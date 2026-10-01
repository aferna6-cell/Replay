"""Train the behaviour-cloned NEXT option scorer (ml/bc_policy.py).

Rows are jsonl or jsonl.gz, one decision per line (the Labeler's labels.v1,
docs/labels_schema.md):

  {"state" | "snapshot": <Snapshot dict>, "game_id": str, "build": int,
   "turn": int, "dp_index": int, "mmr": int | null, "options": [<option dict>, ...],
   "chosen": int, "weight": float,
   "created_ts": int (optional; epoch MILLISECONDS, labels.v2),
   "created_at": str (optional; UTC ISO-8601)}

Rows are streamed one file at a time (labels.v2 writes one <game_id>.jsonl.gz
per game): each file is read, split, encoded, and its raw rows dropped; option
matrices go to an on-disk float32 memmap under --cache-dir (a temp dir by
default, deleted at the end unless --keep-cache), so memory stays bounded by one
file plus the per-row metadata. Results are identical to encoding every row in
memory (tests/test_train_bc_streaming.py). Files are independent: a game split
across several files is still one game for the split, but the advisor baseline
keeps its per-game state per file.

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
fewer than 31 games it refuses: use --pilot. --heldout-ids FILE uses a FROZEN
held-out set as-is (never recomputed or rewritten; ids absent from the data
are reported in the metrics).

--val carves a validation split from the TRAINING games only (the latest 15%,
at least 30, same ordering rule), trains on the rest and records per-epoch
train loss, validation loss and validation top-1/top-3; it never scores the
held-out set. Use it to pick epochs; then train on all training games.

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
  advisor_hand_lines  with --advisor-both: the advisor with its hand-play
                lines too (the column not chosen by --advisor-hand-lines)
  compare       --compare: a BC checkpoint, or the advisor (--compare advisor,
                or a checkpoint path that does not exist yet, e.g. before the
                first ml/policy_net.pt is installed)

With an advisor column present, metrics["gate"] applies the promotion rule:
PASS iff model_coarse overall top-1 >= the better advisor column's, and no
chosen action type with >= 50 held-out decisions has model_coarse top-1 more
than 3.0 points below the better advisor column for that type.

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
import hashlib
import shutil
import sys
import tempfile
import time
from typing import Callable, Dict, List, Optional, Set

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
HEARTBEAT_S = 60


def progress(msg: str) -> None:
    """Timestamped, flushed progress line; stdout is a file under the daily pipeline."""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# --- data -------------------------------------------------------------------
def data_files(patterns: List[str]) -> List[str]:
    """Input files in read order: each pattern's sorted glob (or the pattern)."""
    return [path for pat in patterns for path in (sorted(glob.glob(pat)) or [pat])]


def read_file_rows(path: str) -> List[Dict]:
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_rows(patterns: List[str]) -> List[Dict]:
    """Every row in memory (small sets / tests; main() streams instead)."""
    return [r for path in data_files(patterns) for r in read_file_rows(path)]


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
    """created_ts as given (labels.v2: epoch milliseconds), else created_at
    (ISO 8601) as epoch seconds, else None. Only the ORDER matters (game_order);
    the two units are never mixed within one label set."""
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
def train_model(items, epochs: int, lr: float, batch: int, seed: int,
                on_epoch: Optional[Callable] = None):
    """Deterministic CPU training; restores torch's global settings after.
    on_epoch(epoch, model, history) runs after each epoch (model in eval mode);
    it must not use random state, so it never changes the result."""
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
        n_batches = math.ceil(len(items) / batch)
        for epoch in range(1, epochs + 1):
            model.train()
            total, wsum = 0.0, 0.0
            order = rng.permutation(len(items))
            t0 = beat = time.time()
            for k, start in enumerate(range(0, len(order), batch), 1):
                tensors = pad_batch([items[i] for i in order[start:start + batch]])
                loss = decision_loss(model, *tensors)
                opt.zero_grad()
                loss.backward()
                opt.step()
                w = float(tensors[-1].sum())
                total += loss.item() * w
                wsum += w
                if time.time() - beat >= HEARTBEAT_S:
                    beat = time.time()
                    progress(f"train: epoch {epoch}/{epochs} batch {k}/{n_batches} "
                             f"loss {total / max(wsum, 1e-8):.4f}")
            history.append(round(total / max(wsum, 1e-8), 6))
            progress(f"train: epoch {epoch}/{epochs} done, loss {history[-1]}, "
                     f"{time.time() - t0:.0f}s")
            if on_epoch is not None:
                model.eval()
                with torch.no_grad():
                    on_epoch(len(history), model, history)
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


# --- streaming --------------------------------------------------------------
def _slim_option(o: Dict) -> Dict:
    """What evaluate() reads from an option: type, source, card_id (coarse key)."""
    src = o.get("source")
    if isinstance(src, dict):
        src = {"zone": src.get("zone"), "slot": src.get("slot")}
    return {"type": o.get("type"), "card_id": o.get("card_id"), "source": src}


def _add_counts(dst: Dict, src: Dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict):
            _add_counts(dst.setdefault(k, {}), v)
        else:
            dst[k] = dst.get(k, 0) + v


def scan_file(path: str) -> List[Dict]:
    """The fields game_order / make_heldout read, for every row of one file."""
    return [{k: r.get(k) for k in ("game_id", "build", "created_ts", "created_at")}
            for r in read_file_rows(path)]


def encode_file(path: str, heldout: Set[str], build) -> Dict:
    """Read one file, split it (split_rows), encode its train / held-out rows
    (encode_row) and drop the raw rows. Returns arrays + slim per-row metadata."""
    rows = read_file_rows(path)
    train, held, stats = split_rows(rows, heldout, build)
    games = {}
    for r in rows:
        g = str(r.get("game_id"))
        ts = _row_ts(r) if build_ok(r, build) else None
        prev = games.get(g, (None, False))
        first = ts if prev[0] is None or (ts is not None and ts < prev[0]) else prev[0]
        games[g] = (first, prev[1] or build_ok(r, build))
    out = {"path": path, "stats": stats, "games": games,
           "flags": {"all": sum(1 for r in rows if r.get("flags")), "train": 0, "heldout": 0}}
    for name, part in (("train", train), ("heldout", held)):
        S, O, n_opt, C, W, meta = [], [], [], [], [], []
        merged = chosen_dup = 0
        for r in part:
            s, o, c, idx = encode_row(r, with_groups=True)
            S.append(s)
            O.append(o)
            n_opt.append(o.shape[0])
            C.append(c)
            W.append(row_weight(r, build))
            merged += len(r["options"]) - o.shape[0]
            dup = sum(1 for g in idx if g == idx[r["chosen"]]) > 1
            chosen_dup += dup
            out["flags"][name] += bool(r.get("flags"))
            m = {"game_id": str(r.get("game_id")), "dp_index": r.get("dp_index"),
                 "mmr": r.get("mmr"), "chosen_option": _slim_option(r["options"][r["chosen"]])}
            if name == "heldout":
                interned: Dict[tuple, Dict] = {}     # plays per position share one dict
                slim = []
                for x in r["options"]:
                    sx = _slim_option(x)
                    src = sx["source"] or {}
                    key = (sx["type"], sx["card_id"], src.get("zone"), src.get("slot"))
                    slim.append(interned.setdefault(key, sx))
                m.update(options=slim, chosen=r["chosen"], groups=idx)
            meta.append(m)
        out[name] = {"S": S, "O": np.concatenate(O) if O else None, "n_opt": n_opt,
                     "C": C, "W": W, "meta": meta, "merged": merged,
                     "chosen_dup": chosen_dup,
                     "counts": decision_counts(part)}
    return out


class EncodedSet:
    """Encoded rows: states in RAM, option matrices in an on-disk float32
    memmap (offsets per row), slim metadata per row."""

    def __init__(self, cache_dir: str, name: str):
        self.path = os.path.join(cache_dir, f"{name}_options.f32")
        self._fh = open(self.path, "wb")
        self._S, self.n_opt, self.C, self.W, self.meta = [], [], [], [], []
        self.merged = self.chosen_dup = 0
        self.counts: Dict = {}
        self.S = self.O = self.off = None

    def extend(self, part: Dict) -> None:
        if part["O"] is not None:
            self._fh.write(np.ascontiguousarray(part["O"], dtype=np.float32).tobytes())
        self._S += part["S"]
        self.n_opt += part["n_opt"]
        self.C += part["C"]
        self.W += part["W"]
        self.meta += part["meta"]
        self.merged += part["merged"]
        self.chosen_dup += part["chosen_dup"]
        _add_counts(self.counts, part["counts"])

    def finish(self) -> "EncodedSet":
        self._fh.close()
        self.S = (np.stack(self._S).astype(np.float32) if self._S
                  else np.zeros((0, enc.STATE_DIM), np.float32))
        self._S = []
        self.off = np.zeros(len(self.n_opt) + 1, dtype=np.int64)
        self.off[1:] = np.cumsum(self.n_opt)
        total = int(self.off[-1])
        self.O = (np.memmap(self.path, dtype=np.float32, mode="c",
                            shape=(total, enc.OPTION_DIM)) if total else None)
        for key in ("by_type", "by_kind"):
            self.counts[key] = dict(sorted(self.counts.get(key, {}).items()))
        return self

    def __len__(self) -> int:
        return len(self.C)

    def encoded(self, i: int):
        """(state, options, chosen) exactly as encode_row returned them."""
        return self.S[i], self.O[self.off[i]:self.off[i + 1]], self.C[i]

    def item(self, i: int):
        return self.encoded(i) + (self.W[i],)

    def close(self) -> None:
        self.O = None


def _pool_map(fn, args: List, workers: int):
    """Ordered map; a process pool when workers > 1 (results identical)."""
    if workers <= 1 or len(args) <= 1:
        return map(fn, args)
    return _pool_imap(fn, args, workers)


def _pool_imap(fn, args: List, workers: int):
    from multiprocessing import get_context
    pool = get_context("spawn").Pool(workers, initializer=_worker_init)
    try:
        for res in pool.imap(fn, args):
            yield res
    finally:
        pool.close()
        pool.join()


def _worker_init():
    torch.set_num_threads(1)


def _encode_job(args):
    return encode_file(*args)


def encode_stream(files: List[str], heldout: Set[str], build, cache_dir: str,
                  workers: int = 1):
    """(train EncodedSet, held-out EncodedSet, info) over every file, in order."""
    train, held = EncodedSet(cache_dir, "train"), EncodedSet(cache_dir, "heldout")
    stats: Dict = {}
    flags = {"all": 0, "train": 0, "heldout": 0}
    games: Dict[str, tuple] = {}
    held_files = []
    progress(f"encode: {len(files)} file(s), {workers} worker(s)")
    beat = time.time()
    for n, res in enumerate(_pool_map(_encode_job, [(f, heldout, build) for f in files],
                                      workers), 1):
        _add_counts(stats, res["stats"])
        _add_counts(flags, res["flags"])
        for g, (ts, ok) in res["games"].items():
            pts, pok = games.get(g, (None, False))
            games[g] = (ts if pts is None or (ts is not None and ts < pts) else pts, pok or ok)
        train.extend(res["train"])
        held.extend(res["heldout"])
        if res["heldout"]["C"]:
            held_files.append(res["path"])
        if time.time() - beat >= HEARTBEAT_S or n == len(files):
            beat = time.time()
            progress(f"encode: {n}/{len(files)} files, {len(train)} train rows")
    info = {"stats": stats, "flags": flags, "games": games, "heldout_files": held_files}
    return train.finish(), held.finish(), info


_ADVISOR_SCORER = [None]    # set in-process by run_advisor_stream; loaded once per worker


def _advisor_job(args):
    path, heldout, build, evalnet_path, hand_lines = args
    from ml.advisor_baseline import AdvisorSession, advisor_results, load_scorer
    if _ADVISOR_SCORER[0] is None:
        _ADVISOR_SCORER[0] = load_scorer(evalnet_path)
    _, held, _ = split_rows(read_file_rows(path), heldout, build)
    session = AdvisorSession(_ADVISOR_SCORER[0], hand_lines=hand_lines)
    hits, diag = advisor_results(held, [row_snapshot(r) for r in held], session)
    return [(str(r.get("game_id")), r.get("dp_index")) for r in held], hits, diag


def merge_advisor_diag(diags: List[Dict]) -> Dict:
    """Sum advisor_results diagnostics over files; rates recomputed the same way."""
    out = {"decisions": 0, "errors": 0, "hero_script_lead": 0, "top1_unmappable": {},
           "top1_ambiguous": {}, "top1_moves": {}, "top1_unmappable_by_chosen_type": {},
           "no_moves": 0}
    for d in diags:
        for k in out:
            if isinstance(out[k], dict):
                _add_counts(out[k], d.get(k, {}))
            else:
                out[k] += d.get(k, 0)
    n, moves1 = max(out["decisions"], 1), out["top1_moves"]
    out["top1_unmappable_rate_by_move"] = {
        kd: round(out["top1_unmappable"].get(kd, 0) / c, 4) for kd, c in sorted(moves1.items())}
    out["top1_ambiguous_rate_by_move"] = {
        kd: round(out["top1_ambiguous"].get(kd, 0) / c, 4) for kd, c in sorted(moves1.items())}
    out["top1_unmappable_rate"] = round(
        (sum(out["top1_unmappable"].values()) + out["no_moves"]) / n, 4)
    out["top1_ambiguous_rate"] = round(sum(out["top1_ambiguous"].values()) / n, 4)
    return out


def run_advisor_stream(files: List[str], heldout: Set[str], build, keys: List[tuple],
                       evalnet_path=None, hand_lines: bool = False, required: bool = False,
                       workers: int = 1) -> Dict:
    """run_advisor over the held-out rows of `files`, re-read one file at a time
    (the advisor needs the full snapshot). `keys` = (game_id, dp_index) of the
    held-out rows in order, to check alignment."""
    try:
        from ml.advisor_baseline import load_scorer
        scorer = load_scorer(evalnet_path)
    except Exception as exc:
        return {"enabled": False, "error": f"eval net failed to load: {exc}"}
    if scorer is None:
        path = evalnet_path or os.path.join(REPO, "ml", "eval_net.pt")
        msg = f"no eval net at {path}"
        if required:
            print(f"warning: {msg}; advisor baseline skipped", file=sys.stderr)
        return {"enabled": False, "error": msg}
    _ADVISOR_SCORER[0] = scorer if workers <= 1 else None
    got, hits, diags = [], [], []
    jobs = [(f, heldout, build, evalnet_path, hand_lines) for f in files]
    progress(f"advisor: scoring held-out rows in {len(jobs)} file(s)")
    beat = time.time()
    try:
        for n, (k, h, d) in enumerate(_pool_map(_advisor_job, jobs, workers), 1):
            got += k
            hits += h
            diags.append(d)
            if time.time() - beat >= HEARTBEAT_S or n == len(jobs):
                beat = time.time()
                progress(f"advisor: {n}/{len(jobs)} files, {len(got)} rows")
    finally:
        _ADVISOR_SCORER[0] = None
    if got != list(keys):
        raise RuntimeError("advisor rows are not aligned with the held-out rows")
    return {"enabled": True, "scorer": getattr(scorer, "name", type(scorer).__name__),
            "evalnet_path": evalnet_path or os.path.join(REPO, "ml", "eval_net.pt"),
            "hand_lines": hand_lines, "hits": hits, "mapping": merge_advisor_diag(diags)}


def gate(results: Dict, advisors: List[str], min_n: int = 50, max_drop: float = 0.03) -> Dict:
    """Promotion gate: model_coarse vs the better advisor column, overall and
    per chosen action type (types with >= min_n held-out decisions)."""
    mc = results["model_coarse"]
    cols = [a for a in advisors if a in results]
    best = max(cols, key=lambda a: (results[a]["overall"]["top1"] or 0.0))
    b1 = results[best]["overall"]["top1"] or 0.0
    per_type, fails = {}, []
    for t, v in mc["per_type"].items():
        top = {a: results[a]["per_type"].get(t, {}).get("top1") or 0.0 for a in cols}
        bt = max(top, key=lambda a: top[a])
        d = round((v["top1"] or 0.0) - top[bt], 6)
        gated = v["n"] >= min_n
        ok = not (gated and d < -max_drop)
        per_type[t] = {"n": v["n"], "model_coarse_top1": v["top1"], "best_advisor": bt,
                       "best_advisor_top1": top[bt], "delta": d, "gated": gated, "pass": ok}
        if not ok:
            fails.append(t)
    overall_ok = (mc["overall"]["top1"] or 0.0) >= b1
    return {"rule": f"model_coarse overall top-1 >= best advisor; no type with >= {min_n} "
                    f"decisions more than {max_drop * 100:.1f} pts below the better advisor",
            "advisors": cols,
            "overall": {"model_coarse_top1": mc["overall"]["top1"], "best_advisor": best,
                        "best_advisor_top1": b1,
                        "delta": round((mc["overall"]["top1"] or 0.0) - b1, 6),
                        "pass": overall_ok},
            "per_type": per_type, "failing_types": fails,
            "result": "PASS" if overall_ok and not fails else "FAIL"}


def _sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# --- CLI --------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the behaviour-cloned NEXT option scorer.")
    p.add_argument("--data", nargs="+", required=True, help="jsonl files or globs")
    p.add_argument("--heldout", default=None,
                   help="JSON of held-out game ids, never trained on "
                        "(default results/policy_heldout_ids.json)")
    p.add_argument("--heldout-ids", default=None,
                   help="FROZEN held-out JSON: used as-is, never recomputed or written "
                        "(overrides --heldout / --make-heldout)")
    p.add_argument("--make-heldout", action="store_true",
                   help="create --heldout from the latest 15%% of games (at least 30) "
                        "by created_ts if the file does not exist yet")
    p.add_argument("--pilot", action="store_true",
                   help="plumbing check for small label sets: hold out the latest game, "
                        "write everything under --pilot-dir, never touch the real "
                        "held-out file, never install")
    p.add_argument("--pilot-dir", default=PILOT_DIR, help="output dir for --pilot")
    p.add_argument("--val", action="store_true",
                   help="validation run: carve the latest 15%% (min 30) of TRAINING games, "
                        "train on the rest, record per-epoch val loss / top-1; no held-out "
                        "scoring, never installs")
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
    p.add_argument("--advisor-both", action="store_true",
                   help="also score the other hand-lines variant, as advisor_hand_lines "
                        "(or advisor_no_hand_lines with --advisor-hand-lines)")
    p.add_argument("--train-fit", action="store_true",
                   help="also report model / random top-1 and top-3 on the training rows")
    p.add_argument("--workers", type=int, default=1,
                   help="processes for encoding and the advisor (results identical)")
    p.add_argument("--cache-dir", default=None,
                   help="where encoded option matrices go (default: a temp dir)")
    p.add_argument("--keep-cache", action="store_true", help="do not delete --cache-dir")
    p.add_argument("--install", action="store_true",
                   help="back up ml/policy_net.pt to ml/policy_net.prev.pt, then install the new one")
    return p.parse_args(argv)


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _light_rows(eset: EncodedSet, idx) -> List[Dict]:
    """Rows as evaluate() reads them when groups are not needed."""
    return [{"mmr": eset.meta[i]["mmr"], "options": [eset.meta[i]["chosen_option"]],
             "chosen": 0} for i in idx]


def _held_rows(eset: EncodedSet) -> List[Dict]:
    return [{"game_id": m["game_id"], "dp_index": m["dp_index"], "mmr": m["mmr"],
             "options": m["options"], "chosen": m["chosen"]} for m in eset.meta]


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.pilot:
        if a.install:
            print("error: --pilot never installs (drop --install)", file=sys.stderr)
            return 2
        for path in (a.heldout, a.heldout_ids, a.out, a.metrics):
            if path and (_same_path(path, HELDOUT_PATH) or _same_path(path, LIVE_PATH)
                         or _same_path(path, PREV_PATH)):
                print(f"error: --pilot must not write {path}", file=sys.stderr)
                return 2
        os.makedirs(a.pilot_dir, exist_ok=True)
        a.heldout = a.heldout or os.path.join(a.pilot_dir, "pilot_heldout_ids.json")
        a.out = a.out or os.path.join(a.pilot_dir, "pilot_policy.pt")
    if a.val and a.install:
        print("error: --val never installs (drop --install)", file=sys.stderr)
        return 2
    files = data_files(a.data)
    frozen = a.heldout_ids is not None and not a.pilot
    if frozen:
        if not os.path.isfile(a.heldout_ids):
            print(f"error: frozen held-out file {a.heldout_ids} not found", file=sys.stderr)
            return 2
        a.heldout = a.heldout_ids
    a.heldout = a.heldout or HELDOUT_PATH
    if not frozen and (a.pilot or not os.path.isfile(a.heldout)):
        if not (a.make_heldout or a.pilot):
            print(f"error: {a.heldout} not found (pass --make-heldout to create it)",
                  file=sys.stderr)
            return 2
        lite = [r for part in _pool_map(scan_file, files, a.workers) for r in part]
        try:
            ids, rule = make_heldout(lite, a.build, pilot=a.pilot)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        del lite
        write_heldout(a.heldout, ids, a.build, rule)
        print(f"wrote held-out split {a.heldout} ({rule}): {len(ids)} game(s)")
    heldout = read_heldout(a.heldout)

    own_cache = a.cache_dir is None
    cache = a.cache_dir or tempfile.mkdtemp(prefix="bc_policy_cache_")
    os.makedirs(cache, exist_ok=True)
    try:
        return _run(a, files, heldout, frozen, cache)
    finally:
        if not a.keep_cache:
            import gc
            gc.collect()                       # release memmaps (Windows locks mapped files)
            if own_cache:
                shutil.rmtree(cache, ignore_errors=True)
            else:
                for name in ("train_options.f32", "heldout_options.f32"):
                    try:
                        os.remove(os.path.join(cache, name))
                    except OSError:
                        pass


def _run(a, files, heldout, frozen, cache) -> int:
    train, held, info = encode_stream(files, heldout, a.build, cache, a.workers)
    counts = info["stats"]
    train_games = {m["game_id"] for m in train.meta}
    overlap = sorted(train_games & heldout)
    if overlap:                                # split_rows makes this impossible
        raise SystemExit(f"held-out games leaked into training: {overlap[:5]}")
    if not len(train):
        print("error: no training rows for build", a.build, file=sys.stderr)
        return 2
    seen = info["games"]
    missing = sorted(g for g in heldout if g not in seen)
    held_ts = [seen[g][0] for g in heldout if g in seen and seen[g][0] is not None]

    fit_idx = list(range(len(train)))
    val_info, curve, on_epoch = None, [], None
    if a.val:
        first = [{"game_id": g, "build": a.build, "created_ts": seen[g][0]} for g in train_games]
        try:
            vids, vrule = make_heldout(first, a.build)
        except ValueError as exc:
            print(f"error: --val: {exc}", file=sys.stderr)
            return 2
        vids = set(vids)
        val_idx = [i for i in fit_idx if train.meta[i]["game_id"] in vids]
        fit_idx = [i for i in fit_idx if train.meta[i]["game_id"] not in vids]
        val_items = [train.item(i) for i in val_idx]
        val_rows, val_enc = _light_rows(train, val_idx), [train.encoded(i) for i in val_idx]
        val_info = {"rule": f"{vrule} among TRAINING games only", "val_games": len(vids),
                    "val_rows": len(val_idx), "fit_rows": len(fit_idx),
                    "fit_games": len(train_games) - len(vids)}

        def on_epoch(epoch, model, history):
            tot = wsum = 0.0
            for k in range(0, len(val_items), 256):
                t = pad_batch(val_items[k:k + 256])
                w = float(t[-1].sum())
                tot += decision_loss(model, *t).item() * w
                wsum += w
            ev = evaluate(val_rows, val_enc, {"model": BCPolicy(model).score_encoded})
            pt = {"epoch": epoch, "train_loss": history[-1],
                  "val_loss": round(tot / max(wsum, 1e-8), 6),
                  "val_top1": ev["model"]["overall"]["top1"],
                  "val_top3": ev["model"]["overall"]["top3"]}
            curve.append(pt)
            print(json.dumps(pt), flush=True)

    items = [train.item(i) for i in fit_idx]
    progress(f"train: {len(items)} rows, {a.epochs} epoch(s), batch {a.batch}")
    model, history = train_model(items, a.epochs, a.lr, a.batch, a.seed, on_epoch=on_epoch)
    del items

    date = datetime.date.today().strftime("%Y%m%d")
    out = a.out or os.path.join(REPO, "results", f"policy_net_{date}.pt")
    metrics_path = a.metrics or os.path.splitext(out)[0] + ".metrics.json"
    fit_games = len({train.meta[i]["game_id"] for i in fit_idx})
    save_policy(model, out, {"build": str(a.build), "seed": a.seed, "epochs": a.epochs,
                             "train_rows": len(fit_idx), "train_games": fit_games,
                             "heldout_games": len(heldout), "created": date})
    split = dict(counts, train_rows=len(train), heldout_rows=len(held),
                 train_games=len(train_games), heldout_games=len(heldout),
                 heldout_game_ids=sorted(heldout), overlap=len(overlap),
                 merged_identical_options={"train": train.merged, "heldout": held.merged},
                 chosen_had_duplicate_merged={"train": train.chosen_dup,
                                              "heldout": held.chosen_dup},
                 flagged_rows=info["flags"], heldout_ids_missing_from_data=missing,
                 heldout_created_ts_range=[min(held_ts), max(held_ts)] if held_ts else None)
    metrics = {
        "label": (PILOT_LABEL if a.pilot else
                  "validation run (training games only; held-out not scored)" if a.val
                  else "held-out evaluation"),
        "candidates": "row options (server legal list); identical encodings merged",
        "checkpoint": out, "encoder_version": enc.ENCODER_VERSION,
        "state_dim": enc.STATE_DIM, "option_dim": enc.OPTION_DIM,
        "build": str(a.build), "seed": a.seed, "epochs": a.epochs, "lr": a.lr,
        "batch": a.batch, "heldout_file": a.heldout, "heldout_frozen": frozen,
        "heldout_file_sha256": _sha256(a.heldout),
        "split": split,
        "decisions": {"train": train.counts, "heldout": held.counts},
        "train_loss": history,
    }
    if a.val:
        metrics["validation"] = {"split": val_info, "curve": curve}
        _write_metrics(metrics_path, metrics)
        print(f"validation run: fit rows {len(fit_idx)} ({fit_games} games), val rows "
              f"{val_info['val_rows']} ({val_info['val_games']} games); final val loss "
              f"{curve[-1]['val_loss'] if curve else None}")
        print(f"wrote {out}\nwrote {metrics_path}")
        return 0

    held_rows = _held_rows(held)
    held_enc = [held.encoded(i) for i in range(len(held))]
    held_groups = [m["groups"] for m in held.meta]
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
    keys = [(m["game_id"], m["dp_index"]) for m in held.meta]
    want_advisor = a.baseline_evalnet is not False or compare_advisor
    if want_advisor:
        advisor = run_advisor_stream(info["heldout_files"], heldout, a.build, keys,
                                     a.evalnet_path, a.advisor_hand_lines,
                                     required=bool(a.baseline_evalnet) or compare_advisor,
                                     workers=a.workers)
        if advisor.get("hits") is not None:
            extra["advisor"] = advisor.pop("hits")
            if compare_advisor:
                extra["compare"] = extra["advisor"]
            if a.advisor_both:
                other = not a.advisor_hand_lines
                name = "advisor_hand_lines" if other else "advisor_no_hand_lines"
                alt = run_advisor_stream(info["heldout_files"], heldout, a.build, keys,
                                         a.evalnet_path, other, workers=a.workers)
                if alt.get("hits") is not None:
                    extra[name] = alt.pop("hits")
                advisor = dict(advisor, **{name: alt})
        elif compare_advisor:
            compare["error"] = advisor.get("error", "advisor unavailable")
    progress(f"evaluate: {len(held_rows)} held-out rows")
    results = evaluate(held_rows, held_enc, scorers, extra=extra, groups=held_groups)
    if "compare" in results:
        cm, cc = results["model"]["overall"], results["compare"]["overall"]
        if cm["top1"] is not None and cc["top1"] is not None:
            compare["model_minus_compare"] = {
                "top1": round(cm["top1"] - cc["top1"], 6),
                "top3": round(cm["top3"] - cc["top3"], 6)}
    advisor_cols = [n for n in ("advisor", "advisor_hand_lines", "advisor_no_hand_lines")
                    if n in results]
    metrics.update({"heldout": results, "compare": compare, "advisor_baseline": advisor,
                    "gate": gate(results, advisor_cols) if advisor_cols and len(held) else None})
    if a.train_fit:
        idx = list(range(len(train)))
        metrics["train_fit"] = evaluate(_light_rows(train, idx),
                                        [train.encoded(i) for i in idx], scorers={
                                            "model": BCPolicy(model).score_encoded})
    _write_metrics(metrics_path, metrics)
    train.close()
    held.close()
    m, r = results["model"]["overall"], results["random"]["overall"]
    if a.pilot:
        print(PILOT_LABEL)
    print(f"train rows {len(train)} ({len(train_games)} games), held-out rows "
          f"{len(held)}; held-out top1 {m['top1']} top3 {m['top3']} "
          f"(random {r['top1']} / {r['top3']})")
    for name in ("model_coarse", "advisor", "advisor_hand_lines", "advisor_no_hand_lines",
                 "compare"):
        if name in results:
            c = results[name]["overall"]
            print(f"{name} top1 {c['top1']} top3 {c['top3']}")
    if advisor.get("enabled"):
        d = advisor["mapping"]
        print(f"advisor ({advisor['scorer']}): #1 unmappable {d['top1_unmappable_rate']}, "
              f"ambiguous {d['top1_ambiguous_rate']}")
    elif advisor.get("error"):
        print(f"advisor baseline off: {advisor['error']}")
    if metrics["gate"]:
        g = metrics["gate"]
        print(f"gate {g['result']}" + (f": failing {g['failing_types']}" if g["failing_types"]
                                       else "") + ("" if g["overall"]["pass"] else
                                                   " (overall top-1 below advisor)"))
    if missing:
        print(f"warning: {len(missing)} held-out id(s) not in the data", file=sys.stderr)
    print(f"wrote {out}\nwrote {metrics_path}")
    if a.install:
        install(out)
        print(f"installed {out} -> {LIVE_PATH} (previous kept at {PREV_PATH})")
    return 0


def _write_metrics(path: str, metrics: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
