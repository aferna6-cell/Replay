"""Train the behaviour-cloned NEXT option scorer (ml/bc_policy.py).

Rows are jsonl or jsonl.gz, one decision per line (the Labeler's labels.v1,
docs/labels_schema.md):

  {"state" | "snapshot": <Snapshot dict>, "game_id": str, "build": int,
   "turn": int, "dp_index": int, "mmr": int | null, "options": [<option dict>, ...],
   "chosen": int, "weight": float, "created_ts": float (optional)}

The candidate set of every row is the row's OWN `options` list (the server's
legal options): it is both the softmax set in training and the ranked set in
held-out scoring. encode.legal_options is never used here (it is the live
overlay's generator and still misses options the server lists). Options the
encoder cannot tell apart (identical vectors, e.g. the same spell aimed at two
heroes) are merged into one candidate; the chosen one's group is the target.

Only --build rows (default 253216) train; other builds get weight 0. Games in
the --heldout JSON (default results/policy_heldout_ids.json) are never trained
on; they are the evaluation set. --make-heldout writes that file (if absent)
from the latest 15% of games, at least 30, ordered by the rows' created_ts (or
created_at); if any game lacks both, all games are ordered by game_id. With
fewer than 31 games it refuses: use --pilot.

--pilot is a plumbing check for small label sets (e.g. the 5-game pilot): it
holds out the single latest game, writes the split, checkpoint and metrics
only under --pilot-dir (default results/pilot/), never writes the real
held-out file and never installs.

Held-out scoring skips forced decisions (a single candidate) and reports
MMR-weighted top-1 / top-3 match rate, overall, per chosen action type and per
kind (play split into hand / activate / dark_gift), on the same decisions, for:
  model         the new checkpoint, at full option granularity
  model_coarse  the same ranking scored like the advisor (a hit if the top
                option has the chosen option's type + source slot; target,
                position and Choose One variant ignored)
  random        a random candidate
  advisor       the live eval-net advisor (ml/advisor_baseline.py), mapped to
                the row options by type + card + source slot; on by default
                when an eval net loads (--evalnet-path, default ml/eval_net.pt)
  compare       --compare: a BC checkpoint, or the advisor (--compare advisor,
                or a checkpoint path that does not exist yet, e.g. before the
                first ml/policy_net.pt is installed)

  python -m ml.train_bc_policy --data "data/firestone/labels/v1/*.jsonl.gz" --make-heldout
  python -m ml.train_bc_policy --data ... --compare ml/policy_net.pt --install
  python -m ml.train_bc_policy --data ... --pilot

Writes results/policy_net_<date>.pt plus results/policy_net_<date>.metrics.json.
CPU only, seeded, deterministic.
"""

import argparse
import datetime
import glob
import gzip
import json
import math
import os
import random
import shutil
import sys
from typing import Dict, List, Set

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from hsbg_coach import encode as enc  # noqa: E402
from ml.bc_policy import (BCPolicy, OptionScorer, decision_loss,  # noqa: E402
                          load_bc_policy, pad_batch, save_policy)

DEFAULT_BUILD = "253216"
HELDOUT_FRAC = 0.15
HELDOUT_MIN = 30
LIVE_PATH = os.path.join(REPO, "ml", "policy_net.pt")
PREV_PATH = os.path.join(REPO, "ml", "policy_net.prev.pt")
HELDOUT_PATH = os.path.join(REPO, "results", "policy_heldout_ids.json")
PILOT_DIR = os.path.join(REPO, "results", "pilot")
PILOT_LABEL = ("PLUMBING CHECK on 1 held-out game (--pilot): verifies the pipeline "
               "runs end to end; NOT a model-quality result")


# --- data -------------------------------------------------------------------
def read_rows(patterns: List[str]) -> List[Dict]:
    rows = []
    for pat in patterns:
        for path in (sorted(glob.glob(pat)) or [pat]):
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def build_ok(row: Dict, build) -> bool:
    return str(row.get("build")) == str(build)


def row_weight(row: Dict, build) -> float:
    """Row weight; any build other than the target counts as 0."""
    if not build_ok(row, build):
        return 0.0
    w = row.get("weight")
    return 1.0 if w is None else float(w)


def row_snapshot(row: Dict):
    """labels.v1 rows carry the snapshot as `state`; older rows as `snapshot`."""
    snap = row.get("snapshot")
    return snap if isinstance(snap, dict) else row.get("state")


def invalid_reason(row: Dict):
    """None if the row can train / be scored, else why not."""
    if not isinstance(row_snapshot(row), dict):
        return "no_snapshot"
    opts, c = row.get("options"), row.get("chosen")
    if not isinstance(opts, list) or not opts:
        return "no_options"
    if not isinstance(c, int) or isinstance(c, bool) or not 0 <= c < len(opts):
        return "bad_chosen"
    if not all(isinstance(o, dict) and o.get("type") for o in opts):
        return "bad_option"
    return None


def _valid(row: Dict) -> bool:
    return invalid_reason(row) is None


def _row_ts(row: Dict):
    """created_ts (epoch seconds), else created_at (ISO 8601) as epoch, else None."""
    ts = row.get("created_ts")
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return float(ts)
    at = row.get("created_at")
    if isinstance(at, str) and at:
        try:
            dt = datetime.datetime.fromisoformat(at.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None


def game_order(rows: List[Dict], build):
    """(target-build game ids oldest -> latest, rule used). Ordered by each
    game's earliest created_ts / created_at; if any game has neither, every game
    is ordered by game_id instead."""
    first: Dict[str, object] = {}
    for r in rows:
        g = r.get("game_id")
        if g is None or not build_ok(r, build):
            continue
        g, ts = str(g), _row_ts(r)
        if g not in first:
            first[g] = ts
        elif ts is not None and (first[g] is None or ts < first[g]):
            first[g] = ts
    if first and all(ts is not None for ts in first.values()):
        return sorted(first, key=lambda g: (first[g], g)), "created_ts"
    return sorted(first), "game_id"


def make_heldout(rows: List[Dict], build, frac: float = HELDOUT_FRAC,
                 minimum: int = HELDOUT_MIN, pilot: bool = False):
    """(held-out game ids, rule text). Latest max(minimum, ceil(frac * games))
    target-build games; --pilot: the single latest game. Raises ValueError if a
    real split would leave fewer than one training game."""
    games, order_by = game_order(rows, build)
    if pilot:
        if len(games) < 2:
            raise ValueError(f"--pilot needs at least 2 games, found {len(games)}")
        return games[-1:], f"pilot: latest 1 game by {order_by}"
    n = max(minimum, int(math.ceil(frac * len(games))))
    if n >= len(games):
        raise ValueError(f"only {len(games)} games for build {build}: a real held-out "
                         f"split needs more than {n} (use --pilot for a plumbing check)")
    return games[len(games) - n:], (f"latest {frac:.0%} of games (min {minimum}) "
                                     f"by {order_by}")


def write_heldout(path: str, ids: List[str], build, rule: str = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"build": str(build),
                   "rule": rule or f"latest {HELDOUT_FRAC:.0%} of games (min {HELDOUT_MIN})",
                   "created": datetime.date.today().isoformat(),
                   "game_ids": list(ids)}, fh, indent=1)


def read_heldout(path: str) -> Set[str]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    ids = data.get("game_ids", []) if isinstance(data, dict) else data
    return {str(g) for g in ids}


def split_rows(rows: List[Dict], heldout: Set[str], build):
    """(train rows, held-out rows, counts). Held-out games never train;
    forced decisions (one legal option) are dropped from both."""
    train, held = [], []
    stats = {"rows": len(rows), "other_build": 0, "invalid": 0, "invalid_reasons": {},
             "zero_weight": 0, "forced_train": 0, "forced_heldout": 0}
    for r in rows:
        if not build_ok(r, build):
            stats["other_build"] += 1
            continue
        why = invalid_reason(r)
        if why:
            stats["invalid"] += 1
            stats["invalid_reasons"][why] = stats["invalid_reasons"].get(why, 0) + 1
            continue
        forced = len(r["options"]) < 2
        if str(r.get("game_id")) in heldout:
            if forced:
                stats["forced_heldout"] += 1
            else:
                held.append(r)
        elif forced:
            stats["forced_train"] += 1
        elif row_weight(r, build) <= 0:
            stats["zero_weight"] += 1
        else:
            train.append(r)
    return train, held, stats


def encode_row(row: Dict, with_groups: bool = False):
    """(state vector, option matrix, chosen index) over the row's OWN server
    options. Options with identical vectors are merged (first one kept), so
    the model never has to split mass between candidates it cannot tell apart.
    with_groups=True appends the option -> merged-candidate index list."""
    snap = row_snapshot(row)
    vecs = [enc.encode_option(snap, o) for o in row["options"]]
    group, keep = {}, []
    index = []
    for v in vecs:
        k = v.tobytes()
        if k not in group:
            group[k] = len(keep)
            keep.append(v)
        index.append(group[k])
    if with_groups:
        return enc.encode_state(snap), np.stack(keep), index[row["chosen"]], index
    return enc.encode_state(snap), np.stack(keep), index[row["chosen"]]


def coarse_key(option: Dict) -> tuple:
    """Advisor-granularity identity: type + source slot (card id when the source
    is none, e.g. Dark Gift vs hero power); target / position / variant ignored."""
    src = option.get("source") or {}
    zone = src.get("zone") or "none"
    return (option.get("type"), zone, src.get("slot"),
            option.get("card_id") if zone == "none" else None)


def option_kind(option: Dict) -> str:
    """Chosen-action kind for per-kind metrics: the type, with play split."""
    t = option.get("type", "?")
    if t != "play":
        return t
    if enc.is_dark_gift(option):
        return "play:dark_gift"
    if enc.is_activate(option):
        return "play:activate"
    return "play:hand"


def decision_counts(rows: List[Dict]) -> Dict:
    by_type, by_kind = {}, {}
    for r in rows:
        o = r["options"][r["chosen"]]
        by_type[o.get("type", "?")] = by_type.get(o.get("type", "?"), 0) + 1
        k = option_kind(o)
        by_kind[k] = by_kind.get(k, 0) + 1
    return {"by_type": dict(sorted(by_type.items())), "by_kind": dict(sorted(by_kind.items()))}


# --- training ---------------------------------------------------------------
def train_model(items, epochs: int, lr: float, batch: int, seed: int):
    """Deterministic CPU training; restores torch's global settings after."""
    prev_det, prev_threads = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        model = OptionScorer(enc.STATE_DIM, enc.OPTION_DIM)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        rng = np.random.RandomState(seed)
        history = []
        for _ in range(epochs):
            model.train()
            total, wsum = 0.0, 0.0
            order = rng.permutation(len(items))
            for start in range(0, len(order), batch):
                tensors = pad_batch([items[i] for i in order[start:start + batch]])
                loss = decision_loss(model, *tensors)
                opt.zero_grad()
                loss.backward()
                opt.step()
                w = float(tensors[-1].sum())
                total += loss.item() * w
                wsum += w
            history.append(round(total / max(wsum, 1e-8), 6))
        model.eval()
        return model, history
    finally:
        torch.use_deterministic_algorithms(prev_det)
        torch.set_num_threads(prev_threads)


# --- evaluation -------------------------------------------------------------
def _mmr(row) -> float:
    v = row.get("mmr")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else 0.0


class _Acc:
    def __init__(self):
        self.n, self.w, self.t1, self.t3 = 0, 0.0, 0.0, 0.0

    def add(self, w, t1, t3):
        self.n += 1
        self.w += w
        self.t1 += w * t1
        self.t3 += w * t3

    def out(self) -> Dict:
        if self.w <= 0:
            return {"n": self.n, "top1": None, "top3": None}
        return {"n": self.n, "top1": round(self.t1 / self.w, 6), "top3": round(self.t3 / self.w, 6)}


def evaluate(rows: List[Dict], encoded, scorers: Dict, extra: Dict = None,
             groups: List = None) -> Dict:
    """MMR-weighted top-1/top-3 (rows without an MMR get the median MMR), overall,
    per chosen action type and per kind, for each scorer, the random baseline,
    each `extra` ranker (name -> per-row (top1, top3) hits, e.g. the advisor)
    and, given `groups` (option -> candidate index per row), `<scorer>_coarse`."""
    extra = extra or {}
    mmrs = [m for m in (_mmr(r) for r in rows) if m > 0]
    fallback = float(np.median(mmrs)) if mmrs else 1.0
    names = list(scorers) + ([f"{n}_coarse" for n in scorers] if groups else [])
    names += ["random"] + list(extra)
    accs = {name: {"__all__": _Acc()} for name in names}
    kaccs = {name: {} for name in names}
    for ri, (row, enc_row) in enumerate(zip(rows, encoded)):
        s, o, c = enc_row[:3]
        w = _mmr(row) or fallback
        n = o.shape[0]
        chosen = row["options"][row["chosen"]]
        typ, kind = chosen.get("type", "?"), option_kind(chosen)
        res = {"random": (1.0 / n, min(3, n) / n)}
        for name, fn in scorers.items():
            order = np.argsort(-np.asarray(fn(s, o)), kind="stable")
            rank = int(np.nonzero(order == c)[0][0])
            res[name] = (float(rank == 0), float(rank < 3))
            if groups:
                ck, g = coarse_key(chosen), groups[ri]
                keys = [{coarse_key(row["options"][i]) for i in range(len(g)) if g[i] == grp}
                        for grp in order[:3]]
                res[f"{name}_coarse"] = (float(ck in keys[0]), float(any(ck in k for k in keys)))
        for name, hits in extra.items():
            res[name] = hits[ri]
        for name, (t1, t3) in res.items():
            for key in ("__all__", typ):
                accs[name].setdefault(key, _Acc()).add(w, t1, t3)
            kaccs[name].setdefault(kind, _Acc()).add(w, t1, t3)
    return {name: {"overall": a["__all__"].out(),
                   "per_type": {k: v.out() for k, v in sorted(a.items()) if k != "__all__"},
                   "per_kind": {k: v.out() for k, v in sorted(kaccs[name].items())}}
            for name, a in accs.items()}


def run_advisor(rows: List[Dict], evalnet_path=None, hand_lines: bool = False,
                required: bool = False) -> Dict:
    """Advisor baseline over `rows`: {enabled, scorer, hits, mapping} or
    {enabled: False, error}. Missing eval net = off (an error only if required)."""
    try:
        from ml.advisor_baseline import AdvisorSession, advisor_results, load_scorer
        scorer = load_scorer(evalnet_path)
    except Exception as exc:
        return {"enabled": False, "error": f"eval net failed to load: {exc}"}
    if scorer is None:
        path = evalnet_path or os.path.join(REPO, "ml", "eval_net.pt")
        msg = f"no eval net at {path}"
        if required:
            print(f"warning: {msg}; advisor baseline skipped", file=sys.stderr)
        return {"enabled": False, "error": msg}
    session = AdvisorSession(scorer, hand_lines=hand_lines)
    hits, diag = advisor_results(rows, [row_snapshot(r) for r in rows], session)
    return {"enabled": True, "scorer": getattr(scorer, "name", type(scorer).__name__),
            "evalnet_path": evalnet_path or os.path.join(REPO, "ml", "eval_net.pt"),
            "hand_lines": hand_lines, "hits": hits, "mapping": diag}


def install(path: str, live: str = LIVE_PATH, prev: str = PREV_PATH) -> None:
    """Current live checkpoint -> policy_net.prev.pt, new one -> policy_net.pt."""
    load_bc_policy(path)                      # refuse to install one that won't load
    if os.path.isfile(live):
        shutil.copy2(live, prev)
    shutil.copy2(path, live)


# --- CLI --------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the behaviour-cloned NEXT option scorer.")
    p.add_argument("--data", nargs="+", required=True, help="jsonl files or globs")
    p.add_argument("--heldout", default=None,
                   help="JSON of held-out game ids, never trained on "
                        "(default results/policy_heldout_ids.json)")
    p.add_argument("--make-heldout", action="store_true",
                   help="create --heldout from the latest 15%% of games (at least 30) "
                        "by created_ts if the file does not exist yet")
    p.add_argument("--pilot", action="store_true",
                   help="plumbing check for small label sets: hold out the latest game, "
                        "write everything under --pilot-dir, never touch the real "
                        "held-out file, never install")
    p.add_argument("--pilot-dir", default=PILOT_DIR, help="output dir for --pilot")
    p.add_argument("--build", default=DEFAULT_BUILD, help="game build to train on")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="checkpoint (default results/policy_net_<date>.pt)")
    p.add_argument("--metrics", default=None, help="metrics JSON (default <out>.metrics.json)")
    p.add_argument("--compare", default=None,
                   help="BC checkpoint scored on the same rows, or 'advisor' for the "
                        "eval-net advisor; a path that does not exist (no BC policy "
                        "live yet) also compares against the advisor")
    p.add_argument("--baseline-evalnet", action=argparse.BooleanOptionalAction, default=None,
                   help="score the live eval-net advisor on the held-out rows "
                        "(default: on when an eval net loads)")
    p.add_argument("--evalnet-path", default=None,
                   help="eval net for the advisor baseline (default ml/eval_net.pt; a "
                        "set_net.pt beside it wins, as live); read only")
    p.add_argument("--advisor-hand-lines", action=argparse.BooleanOptionalAction, default=False,
                   help="lead the advisor's ranking with the free-hand-minion 'Play X' "
                        "lines live.advice_lines prints above the ranked actions "
                        "(default off: rank_actions order only)")
    p.add_argument("--install", action="store_true",
                   help="back up ml/policy_net.pt to ml/policy_net.prev.pt, then install the new one")
    return p.parse_args(argv)


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.pilot:
        if a.install:
            print("error: --pilot never installs (drop --install)", file=sys.stderr)
            return 2
        for path in (a.heldout, a.out, a.metrics):
            if path and (_same_path(path, HELDOUT_PATH) or _same_path(path, LIVE_PATH)
                         or _same_path(path, PREV_PATH)):
                print(f"error: --pilot must not write {path}", file=sys.stderr)
                return 2
        os.makedirs(a.pilot_dir, exist_ok=True)
        a.heldout = a.heldout or os.path.join(a.pilot_dir, "pilot_heldout_ids.json")
        a.out = a.out or os.path.join(a.pilot_dir, "pilot_policy.pt")
    a.heldout = a.heldout or HELDOUT_PATH
    rows = read_rows(a.data)
    if a.pilot or not os.path.isfile(a.heldout):
        if not (a.make_heldout or a.pilot):
            print(f"error: {a.heldout} not found (pass --make-heldout to create it)",
                  file=sys.stderr)
            return 2
        try:
            ids, rule = make_heldout(rows, a.build, pilot=a.pilot)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        write_heldout(a.heldout, ids, a.build, rule)
        print(f"wrote held-out split {a.heldout} ({rule}): {len(ids)} game(s)")
    heldout = read_heldout(a.heldout)
    train_rows, held_rows, counts = split_rows(rows, heldout, a.build)
    train_games = {str(r.get("game_id")) for r in train_rows}
    overlap = sorted(train_games & heldout)
    if overlap:                                # split_rows makes this impossible
        raise SystemExit(f"held-out games leaked into training: {overlap[:5]}")
    if not train_rows:
        print("error: no training rows for build", a.build, file=sys.stderr)
        return 2

    items, merged = [], {"train": 0, "heldout": 0}
    for r in train_rows:
        s, o, c = encode_row(r)
        merged["train"] += len(r["options"]) - o.shape[0]
        items.append((s, o, c, row_weight(r, a.build)))
    model, history = train_model(items, a.epochs, a.lr, a.batch, a.seed)

    held_enc = [encode_row(r) for r in held_rows]
    merged["heldout"] = sum(len(r["options"]) - e[1].shape[0]
                            for r, e in zip(held_rows, held_enc))
    held_groups = [encode_row(r, with_groups=True)[3] for r in held_rows]
    scorers = {"model": BCPolicy(model).score_encoded}
    compare = None
    compare_advisor = False
    if a.compare:
        if a.compare.lower() in ("advisor", "evalnet", "eval-net"):
            compare_advisor = True
            compare = {"path": None, "kind": "advisor"}
        elif not os.path.isfile(a.compare):
            compare_advisor = True
            compare = {"path": a.compare, "kind": "advisor",
                       "note": "no BC checkpoint at this path; compared against the "
                               "eval-net advisor instead"}
        else:
            try:
                scorers["compare"] = load_bc_policy(a.compare).score_encoded
                compare = {"path": a.compare, "kind": "bc_checkpoint"}
            except Exception as exc:           # report, don't abort the run
                compare = {"path": a.compare, "kind": "bc_checkpoint", "error": str(exc)}
    extra, advisor = {}, {"enabled": False}
    want_advisor = a.baseline_evalnet is not False or compare_advisor
    if want_advisor:
        advisor = run_advisor(held_rows, a.evalnet_path, a.advisor_hand_lines,
                              required=bool(a.baseline_evalnet) or compare_advisor)
        if advisor.get("hits") is not None:
            extra["advisor"] = advisor.pop("hits")
            if compare_advisor:
                extra["compare"] = extra["advisor"]
        elif compare_advisor:
            compare["error"] = advisor.get("error", "advisor unavailable")
    results = evaluate(held_rows, held_enc, scorers, extra=extra, groups=held_groups)
    if "compare" in results:
        cm, cc = results["model"]["overall"], results["compare"]["overall"]
        if cm["top1"] is not None and cc["top1"] is not None:
            compare["model_minus_compare"] = {
                "top1": round(cm["top1"] - cc["top1"], 6),
                "top3": round(cm["top3"] - cc["top3"], 6)}

    date = datetime.date.today().strftime("%Y%m%d")
    out = a.out or os.path.join(REPO, "results", f"policy_net_{date}.pt")
    metrics_path = a.metrics or os.path.splitext(out)[0] + ".metrics.json"
    save_policy(model, out, {"build": str(a.build), "seed": a.seed, "epochs": a.epochs,
                             "train_rows": len(train_rows), "train_games": len(train_games),
                             "heldout_games": len(heldout), "created": date})
    metrics = {
        "label": PILOT_LABEL if a.pilot else "held-out evaluation",
        "candidates": "row options (server legal list); identical encodings merged",
        "checkpoint": out, "encoder_version": enc.ENCODER_VERSION,
        "state_dim": enc.STATE_DIM, "option_dim": enc.OPTION_DIM,
        "build": str(a.build), "seed": a.seed, "epochs": a.epochs, "lr": a.lr,
        "batch": a.batch, "heldout_file": a.heldout,
        "split": dict(counts, train_rows=len(train_rows), heldout_rows=len(held_rows),
                      train_games=len(train_games), heldout_games=len(heldout),
                      heldout_game_ids=sorted(heldout), overlap=len(overlap),
                      merged_identical_options=merged),
        "decisions": {"train": decision_counts(train_rows),
                      "heldout": decision_counts(held_rows)},
        "train_loss": history, "heldout": results, "compare": compare,
        "advisor_baseline": advisor,
    }
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1)
    m, r = results["model"]["overall"], results["random"]["overall"]
    if a.pilot:
        print(PILOT_LABEL)
    print(f"train rows {len(train_rows)} ({len(train_games)} games), held-out rows "
          f"{len(held_rows)}; held-out top1 {m['top1']} top3 {m['top3']} "
          f"(random {r['top1']} / {r['top3']})")
    for name in ("model_coarse", "advisor", "compare"):
        if name in results:
            c = results[name]["overall"]
            print(f"{name} top1 {c['top1']} top3 {c['top3']}")
    if advisor.get("enabled"):
        d = advisor["mapping"]
        print(f"advisor ({advisor['scorer']}): #1 unmappable {d['top1_unmappable_rate']}, "
              f"ambiguous {d['top1_ambiguous_rate']}")
    elif advisor.get("error"):
        print(f"advisor baseline off: {advisor['error']}")
    print(f"wrote {out}\nwrote {metrics_path}")
    if a.install:
        install(out)
        print(f"installed {out} -> {LIVE_PATH} (previous kept at {PREV_PATH})")
    return 0


if __name__ == "__main__":
    sys.exit(main())