"""Accuracy gate: VOD-reconstructed trajectory vs Power.log ground truth.

Record a game of YOUR OWN with both channels running (the normal `watch`
recorder + a screen recording), ingest the recording with scripts/ingest_vod.py,
then:

    python -m vod.validate data/vods/vod-<id>-g1.jsonl data/game-<ts>.jsonl

Per the spec, ship VOD data into training only when board-content accuracy
clears ~95%; below that, fix the reader (frame rate, model, prompt) first.
Matching is by turn number; per turn we report board-name Jaccard overlap and
exact-match rates for gold / tier / hero health.
"""

import json
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple


def _load(path: str) -> List[Dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _by_turn(records: List[Dict]) -> Dict[int, Dict]:
    """Last state per turn number (mirrors reconstruct's last-read-wins)."""
    got: Dict[int, Dict] = {}
    for r in records:
        state = r.get("state") or {}
        turn = state.get("turn")
        if turn is not None:
            got[int(turn)] = state
    return got


def _board_names(state: Dict) -> Counter:
    return Counter(m.get("name") for m in state.get("board", [])
                   if m.get("name"))


def _jaccard(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 1.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 1.0


def compare(vod: List[Dict], truth: List[Dict]) -> Dict:
    vt, tt = _by_turn(vod), _by_turn(truth)
    common = sorted(set(vt) & set(tt))
    per_turn: List[Tuple[int, float]] = []
    exact = {"gold": 0, "tavern_tier": 0, "hero_health": 0}
    counted = {k: 0 for k in exact}
    for t in common:
        per_turn.append((t, _jaccard(_board_names(vt[t]), _board_names(tt[t]))))
        for key in exact:
            truth_v, vod_v = tt[t].get(key), vt[t].get(key)
            if truth_v is not None and vod_v is not None:
                counted[key] += 1
                exact[key] += int(truth_v == vod_v)
    board_acc: Optional[float] = (
        sum(j for _, j in per_turn) / len(per_turn) if per_turn else None)
    return {
        "turns_compared": len(common),
        "turns_vod_only": sorted(set(vt) - set(tt)),
        "turns_truth_only": sorted(set(tt) - set(vt)),
        "board_accuracy": board_acc,
        "per_turn_board": per_turn,
        "exact": {k: (exact[k] / counted[k] if counted[k] else None)
                  for k in exact},
    }


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    rep = compare(_load(argv[0]), _load(argv[1]))
    print(f"Turns compared: {rep['turns_compared']} "
          f"(vod-only {rep['turns_vod_only']}, "
          f"truth-only {rep['turns_truth_only']})")
    for t, j in rep["per_turn_board"]:
        print(f"  turn {t:2d}  board overlap {j:.2f}")
    for k, v in rep["exact"].items():
        print(f"{k:12s} exact-match: {'n/a' if v is None else f'{v:.0%}'}")
    if rep["board_accuracy"] is None:
        print("No overlapping turns — nothing to grade.")
        return 1
    verdict = "PASS" if rep["board_accuracy"] >= 0.95 else "FAIL (<95%)"
    print(f"Board-content accuracy: {rep['board_accuracy']:.1%} -> {verdict}")
    return 0 if rep["board_accuracy"] >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
