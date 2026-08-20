"""Graded-decision corpus -> chat-format SFT dataset for the Director LoRA.

`hsbg_coach/reviewer.py` (`append_training_examples`) appends one JSON line
per graded decision to `data/train_corpus.jsonl`:
`{"state", "candidates", "chosen", "verdict", "placement", "better"}`.

This module converts that corpus into rows a LoRA fine-tune can train on
directly — `{"messages": [system, user, assistant]}` — using **only**:

  * decisions graded ``"good"`` — the Director's own move, as-is; and
  * decisions graded ``"bad"`` **when the reviewer filled in ``better``** —
    rewritten to the hindsight-preferred move, clearly labeled
    ``"source": "hindsight_rewrite"`` (vs ``"director"`` for the straight
    good examples) so a training run can weight or audit them separately.

Pure stdlib — no torch import anywhere in this file. Training itself never
runs here or on the ThinkPad (spec non-goals); see `scripts/retrain_lora.sh`.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

DEFAULT_CORPUS = os.path.join("data", "train_corpus.jsonl")
DEFAULT_OUT = os.path.join("ml", "sft_dataset.jsonl")

# Direct-inclusion threshold for `min_verdict`. "bad" decisions never qualify
# here — they only enter the dataset via the hindsight-rewrite path below,
# independent of this rank. "unknown" never qualifies — no real move to teach.
_VERDICT_RANK = {"unknown": -1, "bad": -1, "questionable": 1, "good": 2}

SYSTEM_PROMPT = (
    "You are the Hearthstone Battlegrounds Turn Director. Given the current "
    "state and the candidate moves, reply with strict JSON: "
    '{"move": "...", "why": "one-line reason"}.'
)


def _user_content(state: Dict, candidates: List[Dict]) -> str:
    return json.dumps({"state": state, "candidates": candidates},
                      sort_keys=True, separators=(",", ":"))


def _state_key(state: Dict) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


def _sft_row(state: Dict, candidates: List[Dict], move: str, why: str,
            source: str) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(state, candidates)},
            {"role": "assistant", "content": json.dumps(
                {"move": move, "why": why}, separators=(",", ":"))},
        ],
        "source": source,
    }


def build_sft_dataset(corpus_path: str = DEFAULT_CORPUS, out_path: str = DEFAULT_OUT,
                      min_verdict: str = "good") -> Dict[str, int]:
    """Read `corpus_path`, write chat-format SFT rows to `out_path`, dedup
    identical states (first occurrence wins), and return counts. Never
    raises on a missing/empty/malformed corpus — an absent file just yields
    an empty dataset (0 rows), which is the expected state before the first
    reviewed game."""
    counts = {"total_lines": 0, "parsed": 0, "included_good": 0,
             "included_rewritten": 0, "deduped": 0, "skipped": 0, "written": 0}
    min_rank = _VERDICT_RANK.get(min_verdict, _VERDICT_RANK["good"])
    seen_states = set()
    rows: List[Dict] = []

    if os.path.isfile(corpus_path):
        with open(corpus_path, encoding="utf-8") as fh:
            for line in fh:
                counts["total_lines"] += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    counts["skipped"] += 1
                    continue
                counts["parsed"] += 1

                state = rec.get("state") or {}
                candidates = rec.get("candidates") or []
                chosen = rec.get("chosen") or {}
                verdict = rec.get("verdict", "unknown")
                better = (rec.get("better") or "").strip()

                move, why, source = None, None, None
                if verdict != "bad" and _VERDICT_RANK.get(verdict, -1) >= min_rank \
                        and verdict != "unknown":
                    move = chosen.get("move", "")
                    why = chosen.get("why", "")
                    source = chosen.get("source") or "director"
                    label = "good"
                elif verdict == "bad" and better:
                    move = better
                    orig = chosen.get("move", "")
                    why = (f"Hindsight-corrected: '{orig}' was graded bad "
                          f"({rec.get('note', '') or 'lost board equity'}); "
                          f"the better line was '{better}'.")
                    source = "hindsight_rewrite"
                    label = "rewritten"
                else:
                    counts["skipped"] += 1
                    continue

                key = _state_key(state)
                if key in seen_states:
                    counts["deduped"] += 1
                    continue
                seen_states.add(key)

                counts["included_good" if label == "good" else "included_rewritten"] += 1
                rows.append(_sft_row(state, candidates, move, why, source))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    counts["written"] = len(rows)
    return counts


def _cli(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the LoRA SFT dataset from the reviewer's graded corpus.")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help=f"graded-decision JSONL (default: {DEFAULT_CORPUS})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"SFT dataset JSONL to write (default: {DEFAULT_OUT})")
    ap.add_argument("--min-verdict", default="good",
                    choices=sorted(_VERDICT_RANK, key=lambda k: _VERDICT_RANK[k]),
                    help="lowest verdict rank included directly (default: good); "
                         "'bad' decisions only enter via hindsight rewrite")
    args = ap.parse_args(argv)
    counts = build_sft_dataset(args.corpus, args.out, args.min_verdict)
    print(f"SFT dataset: {counts['written']} row(s) written to {args.out} "
          f"(good={counts['included_good']} rewritten={counts['included_rewritten']} "
          f"deduped={counts['deduped']} skipped={counts['skipped']} "
          f"parsed={counts['parsed']}/{counts['total_lines']} corpus lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
