"""Train the behaviour-cloned NEXT option scorer (ml/bc_policy.py).

Rows are jsonl, one decision per line (emitted by the Labeler):

  {"snapshot": <Snapshot dict>, "game_id": str, "build": int, "turn": int,
   "dp_index": int, "mmr": int | null, "options": [<option dict>, ...],
   "chosen": int, "weight": float}

Only --build rows (default 253216) train; other builds get weight 0. Games in
the --heldout JSON are never trained on; they are the evaluation set.
--make-heldout writes that file (if absent) from the most recent 15% of games,
at least 30, where "most recent" means last first-appearance in the --data
files (the Labeler appends games in play order).

Held-out scoring skips forced decisions (a single legal option) and reports
MMR-weighted top-1 / top-3 match rate, overall and per chosen action type, for
the model, a random-legal-move baseline, and optionally a --compare checkpoint.

  python -m ml.train_bc_policy --data data/labels/*.jsonl \
      --heldout data/bc_heldout_games.json --make-heldout
  python -m ml.train_bc_policy --data ... --heldout ... --compare ml/policy_net.pt --install

Writes results/policy_net_<date>.pt plus results/policy_net_<date>.metrics.json.
CPU only, seeded, deterministic.
"""

import argparse
import datetime
import glob
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


# --- data -------------------------------------------------------------------
def read_rows(patterns: List[str]) -> List[Dict]:
    rows = []
    for pat in patterns:
        for path in (sorted(glob.glob(pat)) or [pat]):
            with open(path, encoding="utf-8") as fh:
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


def _valid(row: Dict) -> bool:
    opts, c = row.get("options") or [], row.get("chosen")
    return (isinstance(row.get("snapshot"), dict) and isinstance(c, int)
            and not isinstance(c, bool) and 0 <= c < len(opts))


def game_order(rows: List[Dict], build) -> List[str]:
    seen, order = set(), []
    for r in rows:
        g = r.get("game_id")
        if g is None or not build_ok(r, build) or str(g) in seen:
            continue
        seen.add(str(g))
        order.append(str(g))
    return order


def make_heldout(rows: List[Dict], build, frac: float = HELDOUT_FRAC,
                 minimum: int = HELDOUT_MIN) -> List[str]:
    """Most recent max(minimum, frac * games) target-build games."""
    games = game_order(rows, build)
    n = max(minimum, int(math.ceil(frac * len(games))))
    if n >= len(games):
        n = max(len(games) - 1, 0)          # always leave a game to train on
        print(f"warning: only {len(games)} games; holding out {n}", file=sys.stderr)
    return games[len(games) - n:] if n else []


def write_heldout(path: str, ids: List[str], build) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"build": str(build),
                   "rule": f"most recent {HELDOUT_FRAC:.0%} of games (min {HELDOUT_MIN}) "
                           "by first appearance in the data files",
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
    stats = {"rows": len(rows), "other_build": 0, "invalid": 0, "zero_weight": 0,
             "forced_train": 0, "forced_heldout": 0}
    for r in rows:
        if not build_ok(r, build):
            stats["other_build"] += 1
            continue
        if not _valid(r):
            stats["invalid"] += 1
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


def encode_row(row: Dict):
    snap = row["snapshot"]
    return enc.encode_state(snap), np.stack([enc.encode_option(snap, o) for o in row["options"]])


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


def evaluate(rows: List[Dict], encoded, scorers: Dict) -> Dict:
    """MMR-weighted top-1/top-3 (rows without an MMR get the median MMR), overall
    and per chosen action type, for each scorer plus the random-legal baseline."""
    mmrs = [m for m in (_mmr(r) for r in rows) if m > 0]
    fallback = float(np.median(mmrs)) if mmrs else 1.0
    accs = {name: {"__all__": _Acc()} for name in list(scorers) + ["random"]}
    for row, (s, o) in zip(rows, encoded):
        w = _mmr(row) or fallback
        n, c = o.shape[0], row["chosen"]
        typ = row["options"][c].get("type", "?")
        res = {"random": (1.0 / n, min(3, n) / n)}
        for name, fn in scorers.items():
            order = np.argsort(-np.asarray(fn(s, o)), kind="stable")
            rank = int(np.nonzero(order == c)[0][0])
            res[name] = (float(rank == 0), float(rank < 3))
        for name, (t1, t3) in res.items():
            for key in ("__all__", typ):
                accs[name].setdefault(key, _Acc()).add(w, t1, t3)
    return {name: {"overall": a["__all__"].out(),
                   "per_type": {k: v.out() for k, v in sorted(a.items()) if k != "__all__"}}
            for name, a in accs.items()}


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
    p.add_argument("--heldout", required=True, help="JSON of held-out game ids (never trained on)")
    p.add_argument("--make-heldout", action="store_true",
                   help="create --heldout from the most recent 15%% of games (at least 30) "
                        "if the file does not exist yet")
    p.add_argument("--build", default=DEFAULT_BUILD, help="game build to train on")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="checkpoint (default results/policy_net_<date>.pt)")
    p.add_argument("--metrics", default=None, help="metrics JSON (default <out>.metrics.json)")
    p.add_argument("--compare", default=None, help="another checkpoint scored on the same rows")
    p.add_argument("--install", action="store_true",
                   help="back up ml/policy_net.pt to ml/policy_net.prev.pt, then install the new one")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    rows = read_rows(a.data)
    if not os.path.isfile(a.heldout):
        if not a.make_heldout:
            print(f"error: {a.heldout} not found (pass --make-heldout to create it)",
                  file=sys.stderr)
            return 2
        write_heldout(a.heldout, make_heldout(rows, a.build), a.build)
        print(f"wrote held-out split {a.heldout}")
    heldout = read_heldout(a.heldout)
    train_rows, held_rows, counts = split_rows(rows, heldout, a.build)
    train_games = {str(r.get("game_id")) for r in train_rows}
    overlap = sorted(train_games & heldout)
    if overlap:                                # split_rows makes this impossible
        raise SystemExit(f"held-out games leaked into training: {overlap[:5]}")
    if not train_rows:
        print("error: no training rows for build", a.build, file=sys.stderr)
        return 2

    items = []
    for r in train_rows:
        s, o = encode_row(r)
        items.append((s, o, r["chosen"], row_weight(r, a.build)))
    model, history = train_model(items, a.epochs, a.lr, a.batch, a.seed)

    held_enc = [encode_row(r) for r in held_rows]
    scorers = {"model": BCPolicy(model).score_encoded}
    compare = None
    if a.compare:
        try:
            scorers["compare"] = load_bc_policy(a.compare).score_encoded
            compare = {"path": a.compare}
        except Exception as exc:               # report, don't abort the run
            compare = {"path": a.compare, "error": str(exc)}
    results = evaluate(held_rows, held_enc, scorers)

    date = datetime.date.today().strftime("%Y%m%d")
    out = a.out or os.path.join(REPO, "results", f"policy_net_{date}.pt")
    metrics_path = a.metrics or os.path.splitext(out)[0] + ".metrics.json"
    save_policy(model, out, {"build": str(a.build), "seed": a.seed, "epochs": a.epochs,
                             "train_rows": len(train_rows), "train_games": len(train_games),
                             "heldout_games": len(heldout), "created": date})
    metrics = {
        "checkpoint": out, "encoder_version": enc.ENCODER_VERSION,
        "state_dim": enc.STATE_DIM, "option_dim": enc.OPTION_DIM,
        "build": str(a.build), "seed": a.seed, "epochs": a.epochs, "lr": a.lr,
        "batch": a.batch, "heldout_file": a.heldout,
        "split": dict(counts, train_rows=len(train_rows), heldout_rows=len(held_rows),
                      train_games=len(train_games), heldout_games=len(heldout),
                      overlap=len(overlap)),
        "train_loss": history, "heldout": results, "compare": compare,
    }
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1)
    m, r = results["model"]["overall"], results["random"]["overall"]
    print(f"train rows {len(train_rows)} ({len(train_games)} games), held-out rows "
          f"{len(held_rows)}; held-out top1 {m['top1']} top3 {m['top3']} "
          f"(random {r['top1']} / {r['top3']})")
    if "compare" in results:
        c = results["compare"]["overall"]
        print(f"compare top1 {c['top1']} top3 {c['top3']}")
    print(f"wrote {out}\nwrote {metrics_path}")
    if a.install:
        install(out)
        print(f"installed {out} -> {LIVE_PATH} (previous kept at {PREV_PATH})")
    return 0


if __name__ == "__main__":
    sys.exit(main())