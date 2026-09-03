"""Per-comp playbooks — deep card knowledge the Director draws on for
"bounded experimentation" (specs/llm-turn-director_spec.md req 12) and curve
fluency (req 9).

A playbook is a short markdown file *we write*, in our own words, compiled
from data already in this repo:

  * **Identity** + **Key cards**   — ``final_boards.py``'s real winning-board
    core-card frequencies (falls back to the comp's tribe-strength core cards
    when a comp has no matching board data — same fallback firestone_stats.py
    already uses upstream, restated here so a playbook never needs the raw
    Firestone payload to render).
  * **Enablers**                   — cards whose rules text/keywords generate
    or buff the comp's tribe, found two ways: (1) keyword/text scanning over
    the card KB (``synergy.derive_tags``, same tags the shop-ranker uses) and
    (2) card2vec nearest neighbors of the comp's key cards (co-occurrence the
    text scan can't see — a card with no tribe mention that still wins
    *with* this tribe's key pieces).
  * **Curve**                      — the comp's key cards' average tavern
    tier vs. the measured top-10% pace (``pace.py``): "core pieces average
    tier ~X; top players reach tier X by turn ~T".
  * **How to pilot** + **Pivot signals** — deterministic templates keyed off
    the comp's data-derived tier/placement/frequency, not free text — no
    two runs produce different prose for the same input data, and nothing
    here is copied from a guide site. It IS allowed to be a plain, honest
    restatement of "this comp is S-tier, commit early" — that's the point.

HONEST SCOPE: "How to pilot" and "Pivot signals" are templated guidance, not
learned strategy — a genuinely smarter pilot script is a later ML step
(specs §6). The templates encode conservative, defensible defaults (commit
harder on higher tiers, always leave a pivot window) so nothing here can
actively mislead if the underlying data is thin.

Runs fully offline against the committed snapshots (``data/stats/``,
``data/cards/``) — no network calls, stdlib only.
"""

import os
import re
from typing import Dict, Iterable, List, Optional

from .cards import CardKnowledge, by_name, load_kb
from . import pace as pace_mod
from .stats import CompStats, StatsDB, load_final_boards
from .synergy import _cosine, derive_tags, load_embeddings

DEFAULT_PLAYBOOK_DIR = "data/playbooks"
EXPERIMENTS_FILE = "_experiments.md"

KEY_CARDS_LIMIT = 8
ENABLER_LIMIT = 8
NEIGHBOR_LIMIT = 5
NEIGHBOR_MIN_SIM = 0.2      # below this a card2vec "neighbor" is noise

_PILOT_TEMPLATES = {
    "S": ("Top-tier comp ({avg:.2f} avg placement over {pop:.0%} of games). "
          "Commit once 2+ key cards are open in your first few shops — this "
          "is worth contesting hard."),
    "A": ("Strong comp ({avg:.2f} avg placement). Commit once 2+ key cards "
          "show up, but keep the pivot window open through turn ~6-7 in case "
          "it's contested."),
    "B": ("Middling comp ({avg:.2f} avg placement). Only commit if the key "
          "cards are going uncontested, or your hero/trinkets point this "
          "way — don't force it over a stronger open lane."),
    "C": ("Below-average comp ({avg:.2f} avg placement). Play it "
          "opportunistically off an early hot board, not as a plan A."),
    "D": ("Weak comp by the data ({avg:.2f} avg placement). Treat its "
          "pieces as filler stats, not a build target."),
    "?": ("Untiered — not enough data yet ({avg:.2f} avg placement shown, "
          "low confidence). Treat any pick here like an experiment: tag it "
          "and let the reviewer grade it."),
}


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s or "comp"


def enablers_for_tribe(tribe: Optional[str], kb: Dict[str, CardKnowledge],
                       exclude: Iterable[str] = (), limit: int = ENABLER_LIMIT
                       ) -> List[str]:
    """Cards whose text/keywords generate or buff `tribe` (``buffs:<Tribe>``
    or ``cares:<Tribe>`` synergy tags), sorted cheapest-tier first so the
    earliest-available enablers surface. Shared with ``meta_pack.py`` so the
    lobby-prompt slice and the full playbook agree on what "enabler" means."""
    if not tribe or not kb:
        return []
    excl = {e.lower() for e in exclude}
    # Multiple card_ids can share a display name (reprint/skin variants —
    # e.g. "Arms Dealer" has both a BG26_ and a base RLK_824 id); dedupe by
    # name first so a shared name can't appear twice in one comp's enablers.
    hits: Dict[str, CardKnowledge] = {}
    for c in kb.values():
        if c.name.lower() in excl or c.name in hits:
            continue
        tags = derive_tags(c)
        if f"buffs:{tribe}" in tags or f"cares:{tribe}" in tags:
            hits[c.name] = c
    ordered = sorted(hits.values(),
                     key=lambda c: (c.tier if c.tier is not None else 99, c.name))
    return [c.name for c in ordered[:limit]]


def _nearest_neighbors(names: List[str], emb: Dict[str, List[float]],
                       exclude: Iterable[str], limit: int = NEIGHBOR_LIMIT
                       ) -> List[str]:
    """card2vec co-occurrence neighbors of `names` (comp key cards), pooled
    and ranked by best similarity to any of them — cards that win *with*
    this comp's pieces even when nothing in their text says so."""
    excl = {e.lower() for e in exclude} | {n.lower() for n in names}
    seeds = [emb[n] for n in names if n in emb]
    if not seeds:
        return []
    best: Dict[str, float] = {}
    for cand_name, vec in emb.items():
        if cand_name.lower() in excl:
            continue
        sim = max(_cosine(seed, vec) for seed in seeds)
        if sim >= NEIGHBOR_MIN_SIM:
            best[cand_name] = sim
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:limit]]


def _key_cards(comp: CompStats, board: Optional[dict]) -> List[str]:
    """Real winning-board frequency when we have it, else the comp's
    tribe-strength fallback list (`comp.core_cards`, already computed
    upstream by `firestone_stats.inject_core_cards` for comps with no board
    match)."""
    if board and board.get("coreCards"):
        return [c["name"] for c in board["coreCards"][:KEY_CARDS_LIMIT]]
    return list(comp.core_cards[:KEY_CARDS_LIMIT])


def _curve_line(key_cards: List[str], kb_by_name: Dict[str, CardKnowledge],
                pace_data: Dict[str, Dict[int, float]]) -> str:
    tiers = [kb_by_name[n].tier for n in key_cards
             if n in kb_by_name and kb_by_name[n].tier is not None]
    if not tiers:
        return ("No tier data for this comp's key cards yet — treat leveling "
                "timing as an experiment until real stats arrive.")
    avg_tier = sum(tiers) / len(tiers)
    tavern = pace_data.get("tavern_tier", {})
    target_turn = None
    for turn in sorted(tavern):
        if tavern[turn] >= avg_tier:
            target_turn = turn
            break
    if target_turn is None:
        target_turn = max(tavern) if tavern else None
    if target_turn is None:
        return f"Core pieces average tavern tier ~{avg_tier:.1f}."
    return (f"Core pieces average tavern tier ~{avg_tier:.1f}; the measured "
            f"top-10% pace reaches that tier by turn ~{target_turn}. Level "
            f"toward turn {target_turn}, don't force it earlier just to "
            f"chase these pieces.")


def _pivot_signals(comp_name: str, key_cards: List[str]) -> List[str]:
    piece = " or ".join(key_cards[:2]) if key_cards else "this comp's key cards"
    return [
        f"No {piece} seen by turn 6-7 -> stop forcing {comp_name}, take the "
        f"strongest open lane instead.",
        "Losing combats turn 4-6 despite following the curve -> the pieces "
        "aren't coming together fast enough here; pivot.",
        f"{comp_name} heavily contested (pieces gone from shops by turn 5) "
        f"-> pivot early rather than fighting for scraps.",
    ]


def _render(comp: CompStats, board: Optional[dict], kb: Dict[str, CardKnowledge],
           kb_by_name: Dict[str, CardKnowledge], emb: Dict[str, List[float]],
           pace_data: Dict[str, Dict[int, float]]) -> str:
    key_cards = _key_cards(comp, board)
    key_source = "real winning-board frequency" if (board and board.get("coreCards")) \
        else "tribe-strength fallback (no board data yet for this comp)"
    enablers = enablers_for_tribe(comp.tribe, kb, exclude=key_cards)
    neighbors = _nearest_neighbors(key_cards, emb, exclude=key_cards + enablers)
    tribe_line = comp.tribe if comp.tribe else "Neutral / no single tribe"
    pilot = _PILOT_TEMPLATES.get(comp.tier, _PILOT_TEMPLATES["?"]).format(
        avg=comp.average_position, pop=comp.popularity)

    lines = [
        f"# {comp.name}",
        "",
        "## Identity",
        f"- Tribe: {tribe_line}",
        f"- Tier: {comp.tier}",
        f"- Avg placement: {comp.average_position:.2f}",
        f"- Popularity: {comp.popularity:.1%}",
        "",
        "## Key cards",
        f"_Source: {key_source}._",
        "",
    ]
    if key_cards:
        lines += [f"- {n}" for n in key_cards]
    else:
        lines.append("- (no key-card data for this comp yet)")
    lines += ["", "## Enablers", ""]
    if enablers:
        lines.append("Text/keyword-derived (buff or care about this tribe):")
        lines += [f"- {n}" for n in enablers]
    else:
        lines.append("- (no text-derived enablers found for this tribe)")
    if neighbors:
        lines += ["", "Learned (card2vec neighbors of the key cards):"]
        lines += [f"- {n}" for n in neighbors]
    lines += ["", "## Curve", "", f"- {_curve_line(key_cards, kb_by_name, pace_data)}",
             "", "## How to pilot", "", f"- {pilot}", "", "## Pivot signals", ""]
    lines += [f"- {p}" for p in _pivot_signals(comp.name, key_cards)]
    lines.append("")
    return "\n".join(lines)


def generate_playbooks(kb: Optional[Dict[str, CardKnowledge]] = None,
                       stats_db: Optional[StatsDB] = None,
                       emb: Optional[Dict[str, List[float]]] = None,
                       boards: Optional[List[dict]] = None,
                       out_dir: str = DEFAULT_PLAYBOOK_DIR) -> List[str]:
    """Write one markdown playbook per comp in `stats_db.comps`. Returns the
    written paths. Fully offline: every input defaults to the committed
    snapshot loaders."""
    kb = kb if kb is not None else load_kb()
    stats_db = stats_db or StatsDB.load()
    emb = emb if emb is not None else load_embeddings()
    boards = boards if boards is not None else load_final_boards()
    pace_data = pace_mod.load_pace()

    kb_by_name = by_name(kb)
    boards_by_name = {b.get("name", "").lower(): b for b in boards}

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for comp in stats_db.comps:
        board = boards_by_name.get(comp.name.lower())
        text = _render(comp, board, kb, kb_by_name, emb, pace_data)
        path = os.path.join(out_dir, _slug(comp.name) + ".md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        written.append(path)
    return written


def load_playbook(comp_name: str, dir: str = DEFAULT_PLAYBOOK_DIR) -> Optional[str]:
    path = os.path.join(dir, _slug(comp_name) + ".md")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _section(text: str, header: str) -> str:
    """Body of a `## <header>` markdown section, up to the next `## ` or EOF."""
    marker = f"## {header}"
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    nxt = text.find("\n## ", start)
    body = text[start:nxt if nxt >= 0 else len(text)]
    return body.strip("\n")


def _first_line(block: str) -> str:
    for line in block.splitlines():
        line = line.strip("- ").strip()
        if line and not line.startswith("_Source"):
            return line
    return ""


def playbook_summary_for_prompt(comp_names: List[str],
                                dir: str = DEFAULT_PLAYBOOK_DIR,
                                max_chars: int = 3000) -> str:
    """Compact per-comp summary (identity + how-to-pilot + top pivot signal)
    for the LLM's in-context prompt. Deterministic and priority-ordered —
    comps earlier in `comp_names` are kept first if the budget runs out."""
    blocks = []
    for name in comp_names:
        text = load_playbook(name, dir=dir)
        if not text:
            continue
        identity = _first_line(_section(text, "Identity")) or "(no identity line)"
        pilot = _first_line(_section(text, "How to pilot"))
        pivot = _first_line(_section(text, "Pivot signals"))
        block = f"### {name}\n- {identity}"
        if pilot:
            block += f"\n- Pilot: {pilot}"
        if pivot:
            block += f"\n- Pivot: {pivot}"
        blocks.append(block + "\n")

    out = ""
    for block in blocks:
        if len(out) + len(block) > max_chars:
            break
        out += block
    return out.strip("\n")


def load_validated_experiments(dir: str = DEFAULT_PLAYBOOK_DIR) -> Optional[str]:
    """Promoted-experiment notes (specs req 12) — the reviewer writes
    `_experiments.md`; this only reads it. None if it doesn't exist yet."""
    path = os.path.join(dir, EXPERIMENTS_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return fh.read()
