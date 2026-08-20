"""The LLM Turn Director — one move, one reason, every decision point.

Sits between the deterministic engine (`game_value.rank_actions` /
`turn_search.plan_turn_search` compute every legal action's whole-game
equity) and the overlay panel: `suggest_move` asks a small open-weight model
(`llm_client.LLMClient`) to pick ONE move *from or over* those candidates,
gates the reply through `validator.validate`, and — because the panel must
never show nothing (spec req 1/6) — falls back to the engine's own
top-ranked candidate on any LLM error, timeout, unparseable reply, or
validator rejection.

`log_suggestion` appends one line per decision to `data/suggestions/`; that
file is the between-game reviewer's input (spec req 7) — its schema is a
contract with that (separate) agent, so its keys are not renamed casually.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Set

from .llm_client import LLMClient, LLMError
from .validator import validate

_SUGGESTIONS_DIR_DEFAULT = "data/suggestions"

_GAME_RULES_ESSENTIALS = (
    "Battlegrounds rules: buying a shop minion costs 3 gold and adds it to "
    "your board (max 7, sell one first if full); selling refunds 1 gold; "
    "rolling the shop costs 1 gold for a fresh offer; leveling the tavern "
    "costs the gold shown and raises the tier of minions the shop can offer; "
    "a triple (owning 3 copies of one minion) golds it and grants a "
    "Discover of a higher-tier minion — almost always take a triple over "
    "anything else. Floating unspent gold by ending the turn is a mistake "
    "unless you have infinite economy, are deliberately freezing a shop "
    "you can't afford yet, or are saving gold for next turn's spike — say "
    "which, in 'justification', when you do it on purpose. "
    "Hero power: when the state shows a usable hero power, weigh using it "
    "this turn as a first-class move — timing it well is how each hero is "
    "played optimally. If it needs a target, NAME the target, e.g. "
    "'Use hero power on Brann Bronzebeard'. "
    "Dark gift: when the player has a dark-gift button available, WHEN to "
    "press it is a real decision — weigh the turn's tempo cost against the "
    "gift's value, and suggest 'Use dark gift' on the turn that favors it."
)

# Extra framing injected per spec reqs 10/11 depending on what kind of
# decision this is — hero/trinket/discover picks and dark-gift timing are
# each reasoned about differently than a board buy/sell/roll.
_CHOICE_FRAMING = {
    "hero": (
        "This is a HERO PICK. Population average-placement stats are an "
        "input, never the verdict — weigh how each hero's power/passive "
        "fits the lobby's tribes and the board you'd actually build, not "
        "just the lowest avg placement."
    ),
    "trinket": (
        "This is a TRINKET PICK. Population stats are an input, never the "
        "verdict — weigh board fit (does it buff a tribe/keyword you're "
        "actually running, right now) over raw meta placement."
    ),
    "discover": (
        "This is a DISCOVER. Rank by how each option fits your live board "
        "and the comp you're building toward, not just its raw stats."
    ),
    "dark_gift": (
        "This is a DARK GIFT decision. The choice itself is a discover-"
        "style pick (board fit, not just raw stats), but the harder call is "
        "WHEN to press the button at all: weigh this turn's tempo cost "
        "against the gift's value — a great gift taken on the wrong turn "
        "can still lose the turn's tempo. "
        "# CALIBRATE: no live-game dark-gift log signal exists yet (spec "
        "req 10 — no tracker API for dark gifts), so this timing framing is "
        "a heuristic until real press-turn outcomes are captured."
    ),
}


@dataclass
class Suggestion:
    move: str
    why: str
    source: str                    # "llm" | "engine-fallback"
    experiment: bool = False
    hypothesis: str = ""
    plan_update: str = ""
    latency_s: float = 0.0


# --- prompt building ---------------------------------------------------------

def _build_system_prompt(choice_kind: Optional[str]) -> str:
    lines = [
        "You are the Turn Director for a Hearthstone Battlegrounds coach.",
        "Pick exactly ONE move for the player to make right now, plus a "
        "one-line reason. A separate deterministic validator checks your "
        "answer before the player ever sees it, so just answer honestly — "
        "don't hedge or second-guess the rules yourself.",
        "",
        _GAME_RULES_ESSENTIALS,
        "",
        "Reply with exactly ONE JSON object and nothing else:",
        '{"move": "<one action, e.g. \'Buy Monstrous Macaw\' or \'End turn\'>",',
        ' "why": "<one line, at most 120 characters>",',
        ' "justification": "<only when ending the turn with gold left on '
        'purpose: infinite_economy | deliberate_freeze | save_for_spike>",',
        ' "experiment": <true only for an off-meta line you genuinely '
        "believe in — omit or false otherwise>,",
        ' "hypothesis": "<required when experiment is true: what you '
        'expect to happen and why>",',
        ' "plan_update": "<optional: one line, only if your turn/game plan '
        'changed>"}',
        "",
        "Hard rules:",
        "- Exactly one move.",
        "- Never float gold into 'end turn' without a justification value "
        "above.",
        '- "why" is at most 120 characters.',
        "- Every experiment carries a non-empty hypothesis.",
    ]
    framing = _CHOICE_FRAMING.get(choice_kind)
    if framing:
        lines += ["", framing]
    return "\n".join(lines)


def _compact_minion(m) -> str:
    name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
    atk = m.get("attack") if isinstance(m, dict) else getattr(m, "attack", None)
    hp = m.get("health") if isinstance(m, dict) else getattr(m, "health", None)
    if atk is not None and hp is not None:
        return f"{name} ({atk}/{hp})"
    return str(name)


def _snap_get(snapshot, key, default=None):
    return (snapshot.get(key, default) if isinstance(snapshot, dict)
            else getattr(snapshot, key, default))


def _compact_state(snapshot) -> str:
    board = [_compact_minion(m) for m in (_snap_get(snapshot, "board") or [])]
    shop = [_compact_minion(m) for m in (_snap_get(snapshot, "shop") or [])]
    out = (
        f"Turn {_snap_get(snapshot, 'turn')} · tavern tier "
        f"{_snap_get(snapshot, 'tavern_tier')} · gold "
        f"{_snap_get(snapshot, 'gold')} · hero HP "
        f"{_snap_get(snapshot, 'hero_health')}\n"
        f"Board: {', '.join(board) or '(empty)'}\n"
        f"Shop: {', '.join(shop) or '(empty)'}"
    )
    hero = _snap_get(snapshot, "hero_name") or _snap_get(snapshot, "hero")
    if hero:
        out += f"\nHero: {hero}"
    hp = _snap_get(snapshot, "hero_power")
    if hp:
        name = hp.get("name") if isinstance(hp, dict) else None
        cost = hp.get("cost") if isinstance(hp, dict) else None
        usable = hp.get("usable") if isinstance(hp, dict) else None
        text = hp.get("text") if isinstance(hp, dict) else None
        out += (f"\nHero power: {name or '?'} (cost {cost if cost is not None else '?'}"
                f", {'usable now' if usable else 'not usable yet'})")
        if text:
            out += f" — {text}"
    return out


def _describe_candidate(candidate) -> str:
    a = getattr(candidate, "action", candidate)
    describe = getattr(a, "describe", None)
    if callable(describe):
        try:
            return describe()
        except Exception:
            pass
    return str(a)


def _compact_candidates(candidates, limit: int = 12) -> str:
    lines = []
    for c in (candidates or [])[:limit]:
        desc = _describe_candidate(c)
        placement = getattr(c, "placement", None)
        gain = getattr(c, "gain", None)
        reason = getattr(c, "reason", "") or ""
        tail = ""
        if placement is not None:
            tail = f" [expected finish {placement:.1f}"
            tail += f", {gain:+.2f}]" if gain is not None else "]"
        lines.append(f"- {desc}{tail} — {reason}".rstrip())
    return "\n".join(lines) if lines else "(no legal candidates)"


def _build_user_prompt(snapshot, candidates, meta_ctx: str, plan_ctx: str,
                       lessons: str) -> str:
    parts = [
        "Current state:", _compact_state(snapshot), "",
        "Engine's ranked candidates (its own equity numbers — choose "
        "among or over them, don't just repeat #1 blindly):",
        _compact_candidates(candidates),
    ]
    if meta_ctx:
        parts += ["", "Meta context:", meta_ctx]
    if plan_ctx:
        parts += ["", "Turn/game plan so far:", plan_ctx]
    if lessons:
        parts += ["", "Lessons from past games:", lessons]
    return "\n".join(parts)


# --- liberal JSON extraction --------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json(text: str) -> Optional[dict]:
    """Find the first balanced ``{...}`` in `text` (stripping a ```/```json
    fence if present) and parse it. Models routinely wrap JSON in prose or
    code fences even when asked not to — be liberal in what we accept, strict
    in what we then validate."""
    if not text:
        return None
    fence = _FENCE_RE.search(text)
    body = fence.group(1) if fence else text
    start = body.find("{")
    if start == -1:
        return None
    depth = 0
    end = None
    for i in range(start, len(body)):
        ch = body[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    try:
        parsed = json.loads(body[start:end])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# --- fallback ------------------------------------------------------------

def _fallback_suggestion(candidates, latency_s: float,
                         reason: Optional[str] = None) -> Suggestion:
    """The engine's own top candidate, formatted as a Suggestion. The panel
    must never show nothing (spec req 1/6), so this is the floor every path
    through `suggest_move` lands on when the LLM can't be trusted."""
    if not candidates:
        move, why = "End turn", "no legal actions this turn"
    else:
        top = candidates[0]
        move = _describe_candidate(top)
        why = getattr(top, "reason", None) or "engine's top-ranked move this turn"
    if reason:
        why = f"{why} (LLM fallback: {reason})"
    return Suggestion(move=move, why=why[:240], source="engine-fallback",
                      latency_s=round(latency_s, 3))


# --- public API ------------------------------------------------------------

def suggest_move(snapshot, candidates: List, meta_ctx: str, plan_ctx: str,
                 lessons: str, client: LLMClient,
                 kb_names: Optional[Set[str]] = None,
                 choice_kind: Optional[str] = None) -> Suggestion:
    """Ask the LLM for one move, validate it, fall back to the engine's top
    candidate on any failure. Always returns a Suggestion — never raises for
    an LLM-side problem (LLMError, timeout, garbage JSON, validator
    rejection); those are all handled here so the overlay never has to."""
    system = _build_system_prompt(choice_kind)
    user = _build_user_prompt(snapshot, candidates, meta_ctx, plan_ctx, lessons)

    start = time.monotonic()
    try:
        raw = client.chat(system, user, json_mode=True)
    except LLMError as exc:
        return _fallback_suggestion(candidates, time.monotonic() - start, str(exc))

    latency = time.monotonic() - start
    parsed = _extract_json(raw)
    if parsed is None:
        return _fallback_suggestion(candidates, latency, "unparseable LLM reply")

    accepted, rejection = validate(parsed, snapshot, candidates, kb_names)
    if accepted is None:
        return _fallback_suggestion(candidates, latency, rejection)

    return Suggestion(
        move=str(accepted.get("move") or "").strip(),
        why=str(accepted.get("why") or "").strip()[:240],
        source="llm",
        experiment=bool(accepted.get("experiment", False)),
        hypothesis=str(accepted.get("hypothesis") or "").strip(),
        plan_update=str(accepted.get("plan_update") or "").strip(),
        latency_s=round(latency, 3),
    )


def _snapshot_to_jsonable(snapshot):
    if isinstance(snapshot, dict):
        return snapshot
    if hasattr(snapshot, "__dict__"):
        return dict(snapshot.__dict__)
    return {"repr": repr(snapshot)}


def log_suggestion(suggestion: Suggestion, snapshot, candidates: List,
                   dir: str = _SUGGESTIONS_DIR_DEFAULT) -> str:
    """Append one JSON line to ``{dir}/<YYYY-MM-DD>.jsonl``. This schema is
    CONSUMED BY the between-game reviewer agent (spec req 7) — keep the keys
    exactly as written here; add fields rather than rename/remove them."""
    os.makedirs(dir, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(dir, f"{day}.jsonl")

    cand_rows = [
        {"action": _describe_candidate(c), "detail": getattr(c, "reason", "") or ""}
        for c in (candidates or [])[:5]
    ]
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "turn": _snap_get(snapshot, "turn"),
        "snapshot": _snapshot_to_jsonable(snapshot),
        "candidates": cand_rows,
        "chosen": {
            "move": suggestion.move,
            "why": suggestion.why,
            "experiment": suggestion.experiment,
            "hypothesis": suggestion.hypothesis,
            "source": suggestion.source,
        },
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return path


def format_overlay_line(suggestion: Suggestion) -> str:
    """``"MOVE — why"``, tagged ``[experiment]`` when applicable — the
    one line the overlay panel actually renders (spec req 1)."""
    line = f"{suggestion.move} — {suggestion.why}"
    if suggestion.experiment:
        line += " [experiment]"
    return line
