#!/usr/bin/env python3
"""Distill streamer VOD transcripts into structured strategy knowledge.

Transcripts (from scripts/fetch_vod_transcript.py) hold what no stat carries:
WHY a top player levels, pivots, holds gold, or sells — plus the meta reads
they give while answering chat. This turns each transcript into structured
insights via Claude, then compiles everything into one readable playbook.

    python scripts/distill_transcripts.py                # all new transcripts
    python scripts/distill_transcripts.py data/vods/<id>.txt --force

Output (local-only; data/vods/ is gitignored):
    data/vods/insights/<id>.json     structured insights per VOD
    data/vods/insights.md            compiled playbook across all VODs

Needs the anthropic SDK + credentials (ANTHROPIC_API_KEY or `ant auth login`).
These insights are commentary-derived opinions, NOT ground truth — they tune
your own priors and reading of the meta; they are never fed to the eval net.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

VODS_DIR = REPO_ROOT / "data" / "vods"
INSIGHTS_DIR = VODS_DIR / "insights"
PLAYBOOK = VODS_DIR / "insights.md"

MODEL = "claude-opus-5"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

SCHEMA = {
    "type": "object",
    "properties": {
        "hero": {"type": ["string", "null"]},
        "comp": {"type": ["string", "null"],
                 "description": "final composition played, if identifiable"},
        "placement": {"type": ["integer", "null"]},
        "decision_rules": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "situation": {"type": "string"},
                "action": {"type": "string"},
                "reasoning": {"type": "string"},
                "turn": {"type": ["integer", "null"]},
            },
            "required": ["situation", "action", "reasoning", "turn"],
            "additionalProperties": False},
            "description": "condition -> action -> why, as actually narrated"},
        "card_opinions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "card": {"type": "string"},
                "verdict": {"type": "string",
                            "enum": ["strong", "situational", "weak"]},
                "note": {"type": "string"},
            },
            "required": ["card", "verdict", "note"],
            "additionalProperties": False}},
        "leveling_plan": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "turn": {"type": ["integer", "null"]},
                "action": {"type": "string"},
            },
            "required": ["turn", "action"],
            "additionalProperties": False},
            "description": "narrated tavern-tier timing decisions"},
        "qa_insights": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "answer": {"type": "string"},
            },
            "required": ["question", "answer"],
            "additionalProperties": False},
            "description": "strategy content from chat Q&A moments"},
        "general_principles": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["hero", "comp", "placement", "decision_rules",
                 "card_opinions", "leveling_plan", "qa_insights",
                 "general_principles"],
    "additionalProperties": False,
}

_PROMPT = """\
This is the timestamped transcript of a high-level Hearthstone Battlegrounds
player's commentary over their own gameplay (they also answer chat questions
mid-game). Extract STRATEGY KNOWLEDGE only — the decisions they narrate, the
reasoning behind them, timing conventions, card/comp evaluations, and any
strategy content from chat answers.

Rules:
- Only extract what the player actually says or clearly implies — never
  invent reasoning they didn't give.
- Prefer concrete, conditional rules ("if X on turn N, do Y because Z") over
  vague summaries.
- Use exact English card/hero/comp names when the transcript makes them
  identifiable; skip anything too garbled to identify (auto-captions mangle
  card names — resolve only when confident).
- Ignore banter, donations, and non-gameplay chat.

TRANSCRIPT:
"""


def distill(path: Path, client, model: str, use_fallbacks: list) -> dict:
    import anthropic
    text = path.read_text(encoding="utf-8", errors="replace")
    kwargs = dict(
        model=model, max_tokens=16000,
        messages=[{"role": "user", "content": _PROMPT + text}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    try:
        if use_fallbacks[0] and model.startswith("claude-opus-5"):
            stream_cm = client.beta.messages.stream(
                betas=[_FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            stream_cm = client.messages.stream(**kwargs)
        with stream_cm as stream:
            resp = stream.get_final_message()
    except anthropic.BadRequestError as exc:
        if use_fallbacks[0] and ("fallback" in str(exc).lower()
                                 or "beta" in str(exc).lower()):
            use_fallbacks[0] = False
            return distill(path, client, model, use_fallbacks)
        raise
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined this transcript")
    out = next(b.text for b in resp.content if b.type == "text")
    return json.loads(out)


def build_playbook() -> int:
    """Compile every insights/<id>.json into one readable playbook."""
    docs = []
    for path in sorted(INSIGHTS_DIR.glob("*.json")):
        try:
            docs.append((path.stem, json.loads(path.read_text())))
        except (OSError, ValueError):
            continue
    if not docs:
        return 0
    lines = ["# Streamer playbook (distilled VOD commentary)", "",
             f"Compiled from {len(docs)} transcript(s). Opinions, not ground "
             "truth — cross-check against the stats before adopting.", ""]
    lines.append("## Decision rules")
    for vid, d in docs:
        for r in d.get("decision_rules", []):
            turn = f" (turn {r['turn']})" if r.get("turn") else ""
            lines.append(f"- **{r['situation']}**{turn} → {r['action']} — "
                         f"{r['reasoning']} `[{vid}]`")
    lines.append("")
    lines.append("## Leveling timings")
    for vid, d in docs:
        for l in d.get("leveling_plan", []):
            turn = f"turn {l['turn']}: " if l.get("turn") else ""
            lines.append(f"- {turn}{l['action']} `[{vid}]`")
    lines.append("")
    lines.append("## Card reads")
    for vid, d in docs:
        for c in d.get("card_opinions", []):
            lines.append(f"- **{c['card']}** — {c['verdict']}: {c['note']} "
                         f"`[{vid}]`")
    lines.append("")
    lines.append("## Chat Q&A")
    for vid, d in docs:
        for q in d.get("qa_insights", []):
            lines.append(f"- Q: {q['question']}  \n  A: {q['answer']} `[{vid}]`")
    lines.append("")
    lines.append("## General principles")
    for vid, d in docs:
        for p in d.get("general_principles", []):
            lines.append(f"- {p} `[{vid}]`")
    PLAYBOOK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(docs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("transcripts", nargs="*", type=Path,
                    help="transcript .txt files (default: all in data/vods/)")
    ap.add_argument("--force", action="store_true",
                    help="re-distill transcripts that already have insights")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    try:
        import anthropic
    except ImportError:
        print("anthropic SDK missing: pip install -r requirements-vod.txt")
        return 1
    targets = args.transcripts or sorted(VODS_DIR.glob("*.txt"))
    if not targets:
        print("No transcripts in data/vods/ — fetch one first:\n"
              "  python scripts/fetch_vod_transcript.py <vod-url>")
        return 1
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()
    use_fallbacks = [True]
    done = 0
    for t in targets:
        out = INSIGHTS_DIR / f"{t.stem}.json"
        if out.exists() and not args.force:
            continue
        print(f"Distilling {t.name}…")
        try:
            insights = distill(t, client, args.model, use_fallbacks)
        except Exception as exc:
            print(f"  WARN: {exc}")
            continue
        out.write_text(json.dumps(insights, indent=1) + "\n", encoding="utf-8")
        rules = len(insights.get("decision_rules", []))
        print(f"  -> {out.name}: {rules} decision rules, "
              f"{len(insights.get('card_opinions', []))} card reads")
        done += 1
    n = build_playbook()
    if n:
        print(f"Playbook rebuilt from {n} VOD(s) -> {PLAYBOOK}")
    return 0 if (done or n) else 1


if __name__ == "__main__":
    raise SystemExit(main())
