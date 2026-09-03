"""The lobby-start game plan — L3 in the Turn Director architecture
(``specs/llm-turn-director_spec.md``, "L3 Game Plan").

At lobby start the Director gets a ~5s budget to turn the lobby's tribe set
into a strategy memo: a preferred comp (plan A), a fallback (plan B), the
cards that enable each, and the concrete triggers that mean "stop forcing
plan A." That memo is state the Director carries silently and re-checks
every turn (req 5) — the overlay only ever shows one move, but the move is
chosen against this plan.

Flexible tribes (req 5): a `GamePlan` is a *preference*, never a lock. The
lobby's actual shop offers decide the real line; this module says so
explicitly in `notes` so nothing downstream mistakes a plan for a commitment.

Ranking is deterministic and offline: it reads a `meta_pack` dict (built by
``meta_pack.py``) plus optional `tier7_hero_rows` — hero-pick rows carry
lobby+hero-conditioned `top_comps`, the most specific signal available, so
when supplied they win the tie-break over the meta pack's own (already
Tier7-preferring, per `meta_pack.py`) comp ranking.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

DEFAULT_GAME_PLAN_PATH = "data/game_plan.json"

LOW_HP_TRIGGER = 15
LEVEL_BEHIND_TURN = 7


@dataclass
class GamePlan:
    lobby_tribes: List[str]
    plan_a: str
    plan_a_enablers: List[str]
    plan_b: str
    plan_b_enablers: List[str]
    pivot_triggers: List[str]
    notes: str
    revised: List[str] = field(default_factory=list)
    # The REAL winning boards from HDT/Firestone data — the destination the
    # Director paths toward (grounding req 15: the model never invents what
    # wins, it only reasons about how to get there from the current board).
    plan_a_target: List[str] = field(default_factory=list)
    plan_b_target: List[str] = field(default_factory=list)
    # The full ranked watchlist (owner, 2026-08-20: "I don't want plan A and
    # plan B, that is too rigid"): every viable comp this lobby, best first,
    # each with enablers + target board. plan_a/plan_b above are just the
    # top two entries kept as internal shorthands — the Director is shown
    # the whole ranked list and told to prioritize, never force.
    options: List[dict] = field(default_factory=list)


_FINAL_BOARDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                                  "stats", "firestone_final_boards.json")


def _target_board(comp: dict, final_boards: Optional[List[dict]] = None) -> List[str]:
    """The comp's destination board, from a real top-MMR winning example
    (data-driven, position order, deduped); falls back to core cards."""
    if final_boards is None:
        try:
            with open(_FINAL_BOARDS_PATH, encoding="utf-8") as fh:
                final_boards = json.load(fh).get("boards", [])
        except (OSError, ValueError):
            final_boards = []
    want = (comp.get("name") or "").lower()
    for b in final_boards or []:
        if (b.get("name") or "").lower() == want or \
                (b.get("archetype") or "").replace("_", " ").lower() == want:
            examples = b.get("examples") or []
            if examples:
                best = max(examples, key=lambda e: e.get("mmr") or 0)
                seen, names = set(), []
                for m in sorted(best.get("minions", []),
                                key=lambda m: m.get("pos") or 0):
                    n = m.get("name")
                    if n and n not in seen:
                        seen.add(n)
                        names.append(n)
                if names:
                    return names
            if b.get("coreCards"):
                return list(b["coreCards"])
    return list(comp.get("core_cards") or [])


def _candidate_comps(lobby_tribes: List[str], meta_pack: dict) -> List[dict]:
    lobby = {t.lower() for t in lobby_tribes}
    cands = [c for c in meta_pack.get("comps", [])
            if (c.get("tribe") or "").lower() in lobby]
    if not cands:
        # No comp in the pack matches this lobby's tribes (thin data, or a
        # lobby of tribes the pack hasn't seen) — fall back to the pack's
        # overall best comps rather than returning an empty plan.
        cands = list(meta_pack.get("comps", []))
    return cands


def _hero_boost(cands: List[dict], tier7_hero_rows: Optional[List[dict]]
                ) -> Dict[str, int]:
    """How many of the offered heroes' Tier7 `top_comps` name this comp —
    the lobby+hero-conditioned signal, when we have it."""
    boost: Dict[str, int] = {}
    for row in tier7_hero_rows or []:
        for name in row.get("top_comps") or []:
            boost[name.lower()] = boost.get(name.lower(), 0) + 1
    return boost


def _rank(cands: List[dict], boost: Dict[str, int]) -> List[dict]:
    def key(c):
        b = boost.get(c["name"].lower(), 0)
        avg = c.get("avg_placement")
        avg = avg if avg is not None else 9.0
        return (-b, avg)                     # higher boost first, then lower avg
    return sorted(cands, key=key)


def build_game_plan(lobby_tribes: List[str], meta_pack: dict,
                    hero_offer: Optional[List[str]] = None,
                    tier7_hero_rows: Optional[List[dict]] = None) -> GamePlan:
    cands = _candidate_comps(lobby_tribes, meta_pack)
    boost = _hero_boost(cands, tier7_hero_rows)
    ranked = _rank(cands, boost)

    if not ranked:
        return GamePlan(
            lobby_tribes=list(lobby_tribes), plan_a="", plan_a_enablers=[],
            plan_b="", plan_b_enablers=[],
            pivot_triggers=["No comp data available this lobby — play "
                           "fundamentals (stats, curve) and reassess turn 4-5."],
            notes="No meta pack comp data matched this lobby. Plan is a "
                 "placeholder, not a commitment.",
        )

    a = ranked[0]
    b = ranked[1] if len(ranked) > 1 else ranked[0]
    plan_a, plan_b = a.get("name", ""), b.get("name", "")
    plan_a_enablers = list(a.get("enabler_cards") or [])
    plan_b_enablers = list(b.get("enabler_cards") or [])

    core_a = a.get("core_cards") or []
    piece = " or ".join(core_a[:2]) if core_a else f"{plan_a}'s core pieces"
    measured = meta_pack.get("curve", {}).get("measured", {})
    bench = measured.get(LEVEL_BEHIND_TURN) or measured.get(str(LEVEL_BEHIND_TURN))
    pivot_triggers = [
        f"No {piece} by turn {LEVEL_BEHIND_TURN} -> pivot to {plan_b}.",
        f"Below ~{LOW_HP_TRIGGER} HP before turn 8 -> fall back to a tempo "
        f"line (whatever board is strongest), delay leveling.",
        f"{plan_a} heavily contested (pieces gone from shops by turn 5-6) "
        f"-> pivot to {plan_b} early rather than fight for scraps.",
    ]
    if bench is not None:
        pivot_triggers.append(
            f"Tavern tier under ~{bench:.1f} by turn {LEVEL_BEHIND_TURN} "
            f"(measured top-10% pace) two turns running -> prioritize "
            f"leveling over greedy buys, on either plan.")

    boosted = [c["name"] for c in ranked if boost.get(c["name"].lower())]
    hero_note = (f" Hero offer {hero_offer} — Tier7 hero-pick data favors "
                f"{', '.join(boosted[:2])} for this lobby." if hero_offer and boosted
                else "")
    notes = (f"This is a ranked WATCHLIST, not a script — every listed comp "
             f"is live, the ranking is only priority (flexible tribes, spec "
             f"req 5). Commit late, play what the shops actually offer, move "
             f"freely down (or back up) the list when the offers say so, and "
             f"re-check pivot_triggers every turn.{hero_note}")

    options = [{
        "name": c.get("name", ""),
        "tribe": c.get("tribe"),
        "avg_placement": c.get("avg_placement"),
        "enablers": list(c.get("enabler_cards") or []),
        "target": _target_board(c),
    } for c in ranked[:4]]

    return GamePlan(
        lobby_tribes=list(lobby_tribes), plan_a=plan_a,
        plan_a_enablers=plan_a_enablers, plan_b=plan_b,
        plan_b_enablers=plan_b_enablers, pivot_triggers=pivot_triggers,
        notes=notes,
        plan_a_target=_target_board(a), plan_b_target=_target_board(b),
        options=options,
    )


def plan_to_prompt(plan: GamePlan, max_chars: int = 1800) -> str:
    lines = [
        f"Game plan — lobby tribes: {', '.join(plan.lobby_tribes)}",
        "Comp priorities this lobby (ranked from HDT data — prioritize, "
        "never force; the shops decide what you actually play):",
    ]
    options = plan.options or [
        o for o in ({"name": plan.plan_a, "enablers": plan.plan_a_enablers,
                     "target": plan.plan_a_target, "avg_placement": None},
                    {"name": plan.plan_b, "enablers": plan.plan_b_enablers,
                     "target": plan.plan_b_target, "avg_placement": None})
        if o["name"]]
    for i, o in enumerate(options, 1):
        avg = o.get("avg_placement")
        row = f"  {i}. {o.get('name')}" + (f" (avg {avg})" if avg is not None else "")
        if o.get("enablers"):
            row += f" — enablers: {', '.join(o['enablers'])}"
        lines.append(row)
        if o.get("target"):
            lines.append("     TARGET BOARD (real HDT winner — path toward "
                         "it): " + ", ".join(o["target"]))
    lines.append("Pivot triggers:")
    lines += [f"  - {p}" for p in plan.pivot_triggers]
    lines.append(f"Notes: {plan.notes}")
    if plan.revised:
        lines.append("Revisions: " + " | ".join(plan.revised))

    out = ""
    for line in lines:
        candidate = out + line + "\n"
        if len(candidate) > max_chars:
            break
        out = candidate
    return out.rstrip("\n")


def save_plan(plan: GamePlan, path: str = DEFAULT_GAME_PLAN_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(plan), fh, indent=1)
    return path


def load_plan(path: str = DEFAULT_GAME_PLAN_PATH) -> GamePlan:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("revised", [])
    data.setdefault("plan_a_target", [])
    data.setdefault("plan_b_target", [])
    data.setdefault("options", [])
    return GamePlan(**data)


def revise_plan(plan: GamePlan, revision: str) -> GamePlan:
    """Appends `revision` to `plan.revised` (in place) and returns `plan` —
    the Director must say when it revises the plan (spec: "Director may
    revise it and must say so")."""
    plan.revised.append(revision)
    return plan
