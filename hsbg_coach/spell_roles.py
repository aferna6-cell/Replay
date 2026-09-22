"""Tavern (Battlegrounds) spell knowledge + value.

Tavern spells sit in the shop next to minions and cost a variable amount of gold
(a COST tag, unlike minions' flat 3). To recommend one we need to know what it
does — our minion card KB doesn't carry spells — so this module holds a small,
extensible table keyed by cardId, plus a generic fallback so an unknown spell is
still surfaced (named + costed) rather than ignored.

Value is expressed as a *placement adjustment* (negative = better finish) so it
slots straight into the whole-game ranker, same as minions/tech. Known strong
effects (gold, triples, free minions) get a real bonus; unknown spells get a
small, honest "it's an option — read the effect" nudge scaled by how cheap and
affordable they are.

Seed the table as real cardIds are confirmed from live logs (entityName is in the
Power.log, so names resolve even before an effect is curated here).
"""

from typing import Dict, Optional, Tuple

# cardId -> (placement_bonus, note). Bonus is negative = recommend more.
# Seeded conservatively; extend as spells are confirmed from real games.
_KNOWN: Dict[str, Tuple[float, str]] = {
    # "Pointy Arrow" token seen in a real log — minor combat trick, situational.
    "EBG_Spell_014": (-0.1, "tavern spell — small combat trick"),
    # Aberration / Jeef VOD — Corrupted Coin (BG36_303) appears often in frames/ASR.
    "BG36_303": (-0.35, "Corrupted Coin — Jeef/Aberration VOD priority spell"),
    # Calibrated from Power.log shop BATTLEGROUND_SPELL (2026-09 slices). Mild
    # bonuses only — effects not fully labeled; prefer over unknown demotion.
    "BG28_800": (-0.2, "Careful Investment — shop spell (Power.log)"),
    "BG28_503": (-0.15, "Fortify — shop spell (Power.log)"),
    "BG28_520": (-0.1, "Tricky Trousers — shop spell (Power.log)"),
    "BG28_825": (-0.15, "Defender's Rites — shop spell (Power.log)"),
    "BG28_838": (-0.15, "Perfect Vision — shop spell (Power.log)"),
    "BG28_888": (-0.15, "Misplaced Tea Set — shop spell (Power.log)"),
    "BG28_886": (-0.15, "Staff of Enrichment — shop spell (Power.log)"),
    "BG28_571": (-0.15, "Hasty Excavation — shop spell (Power.log)"),
    "BG28_606": (-0.1, "Spitescale Special — shop spell (Power.log)"),
    "BG28_512": (-0.1, "Enchanted Lasso — shop spell (Power.log)"),
    "BG28_521": (-0.1, "Planar Telescope — shop spell (Power.log)"),
    "BG28_884": (-0.1, "Overconfidence — shop spell (Power.log)"),
    "BG28_GIL_836": (-0.1, "Hired Headhunter — shop spell (Power.log)"),
    "BG30_804": (-0.15, "Robust Evolution — shop spell (Power.log)"),
    "BG31_819": (-0.1, "Temperature Shift — shop spell (Power.log)"),
    "BG31_886": (-0.15, "Forest's Bounty — shop spell (Power.log)"),
    "BG32_815": (-0.15, "Shifting Tide — shop spell (Power.log)"),
    "BG33_101": (-0.1, "A New Sprout — shop spell (Power.log)"),
    "BG34_330": (-0.15, "Search Through Time — shop spell (Power.log)"),
    "BG34_689": (-0.15, "Blood Gem Barrage — shop spell (Power.log)"),
    "BG35_922": (-0.15, "Queen's Command — shop spell (Power.log)"),
    "BG36_246": (-0.15, "Mighty Dragonbreath — shop spell (Power.log)"),
    "BG36_884": (-0.15, "Weapons Forge — shop spell (Power.log)"),
    "EBG_Spell_037": (-0.1, "Unmasked Identity — shop spell (Power.log)"),
}

# Generic value when we don't know the spell yet: a cheap, affordable tavern spell
# An unknown spell is NOT a recommendation — we don't know its effect, so we must
# not push it over a real minion buy. Demote it (positive = worse finish) so it only
# surfaces if there's genuinely nothing better; known-good spells still get promoted.
_GENERIC_BONUS = 0.5
_GENERIC_NOTE = "tavern spell (effect unknown — only if nothing better)"


def spell_value(card_id: Optional[str], name: Optional[str], cost: int,
                gold: int) -> Tuple[float, str]:
    """(placement_adjustment, reason) for buying a tavern spell.

    Negative adjustment = better finish. Unknown spells get a modest, affordability
    -scaled nudge so they surface as a real option without pretending to know the
    effect."""
    if card_id and card_id in _KNOWN:
        return _KNOWN[card_id]
    # Unknown: only nudge it up if you can comfortably afford it (cheap relative to
    # your gold), else it's neutral so it doesn't crowd out a real board buy.
    if cost <= 2 and gold >= cost:
        return _GENERIC_BONUS, _GENERIC_NOTE
    return 0.0, _GENERIC_NOTE


def spell_name(card_id: Optional[str], name: Optional[str]) -> str:
    return name or card_id or "Tavern spell"
