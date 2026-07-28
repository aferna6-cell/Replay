"""Card effects for the combat sim — the data layer behind deathrattles etc.

In an auto-battle, three effect categories actually fire (battlecries are a
*recruit-phase* event — their stat result is already on the board the log gives
us, so the sim doesn't re-run them):

- ``Summon`` — a deathrattle that summons tokens at the dead minion's spot.
  Covers the big family: Harvest Golem, Mecharoo, Rat Pack, Scallywag (whose
  token attacks immediately), Spawn-of-N'Zoth-style bodies, and Reborn.
- ``StartOfCombat`` — deal damage to random enemies before the first attack.
  Covers Red Whelp / Prophet-style openers.
- Keyword flags (Windfury, Cleave, Divine Shield, Taunt, Poisonous, Reborn) live
  directly on ``Combatant``.

This registry holds a *representative* set keyed by minion name. Full coverage is
a data problem — populate from HearthstoneJSON / BG JSON, or bridge to Firestone's
open-source simulator (see specs §6). The engine in sim.py is complete; only the
card list is partial, and that's intentional and documented.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Summon:
    count: int
    attack: int
    health: int
    taunt: bool = False
    divine_shield: bool = False
    poisonous: bool = False
    reborn: bool = False
    attack_immediately: bool = False   # Scallywag's Sky Pirate
    name: str = "Token"


@dataclass
class StartOfCombat:
    damage: int
    targets: int = 1                   # number of random enemies hit
    name: str = ""


@dataclass
class DeathBurst:
    """A deathrattle that deals damage to enemies (board-clear tech). hits_all
    True = every enemy (Tunnel Blaster: 3 to all); else `targets` random enemies."""
    damage: int
    hits_all: bool = True
    targets: int = 1


@dataclass
class CardEffects:
    deathrattle: Optional[Summon] = None
    start_of_combat: Optional[StartOfCombat] = None
    death_burst: Optional[DeathBurst] = None
    grants: tuple = ()                 # extra keywords the card itself has, e.g. ("poisonous",)


# Representative registry (keyed by minion name; extend from card data).
# Stats reflect base (un-tripled) tokens. Golden/triple handling is a later pass.
REGISTRY: Dict[str, CardEffects] = {
    "Mecharoo":      CardEffects(deathrattle=Summon(1, 1, 1, name="Jo-E Bot")),
    "Harvest Golem": CardEffects(deathrattle=Summon(1, 2, 1, name="Damaged Golem")),
    "Kaboom Bot":    CardEffects(deathrattle=Summon(0, 0, 0)),  # placeholder: dmg DR (see note)
    "Rat Pack":      CardEffects(deathrattle=Summon(3, 1, 1, name="Rat")),  # ~attack-scaled
    "Scallywag":     CardEffects(deathrattle=Summon(1, 1, 1, attack_immediately=True,
                                                    name="Sky Pirate")),
    "Spawn of N'Zoth": CardEffects(deathrattle=Summon(0, 0, 0)),  # buffs others (later)
    "Red Whelp":     CardEffects(start_of_combat=StartOfCombat(damage=1, targets=1)),
    "Prophet of the Boar": CardEffects(start_of_combat=StartOfCombat(damage=2, targets=1)),
    # Situational tech (modelled so the sim values them by the actual matchup):
    "Tunnel Blaster": CardEffects(death_burst=DeathBurst(damage=3, hits_all=True)),
    "Deadly Spore":   CardEffects(grants=("poisonous",)),   # Venomous ~= poisonous in-sim
}


def effects_for(name: Optional[str], card_id: Optional[str] = None) -> Optional[CardEffects]:
    """Look up effects by minion name (card_id support added once a card map exists)."""
    if name and name in REGISTRY:
        return REGISTRY[name]
    return None
