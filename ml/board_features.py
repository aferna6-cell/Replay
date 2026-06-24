"""Encode a Battlegrounds board into a fixed-length feature vector.

This is the input layer of the deep "board brain." One encoder serves BOTH
training sources so the learned model transfers cleanly:

  * population final boards (Firestone comp data — minions as cardID + tags), and
  * your own live/recorded snapshots (MinionView dicts: name/attack/health/tags).

Features per board:
  * card2vec mean vector  — the learned card *relationships* (synergy geometry).
  * tribe histogram       — comp identity (how Murloc / Mech / Dragon… it is).
  * keyword/stat scalars  — board size, atk/health totals, max tier, golden /
                            divine-shield / reborn / taunt counts.

Hero and trinkets are handled by the model (hero as a learned embedding); this
module only builds the board-derived dense vector. numpy-only (no torch) so it's
reusable wherever a board needs scoring.
"""

from typing import Dict, List, Optional

import numpy as np

from hsbg_coach.cards import CardKnowledge

# Canonical BG tribes (stable index for the histogram).
TRIBES = ["Beast", "Mech", "Murloc", "Dragon", "Demon", "Elemental",
          "Pirate", "Naga", "Undead", "Quilboar", "All"]
_TRIBE_IX = {t: i for i, t in enumerate(TRIBES)}

# dense layout = [card2vec mean | tribe hist | scalars]; scalars listed for docs.
_SCALARS = ["size", "sum_atk", "sum_hp", "mean_atk", "mean_hp", "max_tier",
            "mean_tier", "golden", "divine", "reborn", "taunt"]


def _i(v, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _minion(name: str, atk: int, health: int, tier: int, golden: bool,
            divine: bool, reborn: bool, taunt: bool, tribes: List[str]) -> Dict:
    return dict(name=name, atk=atk, health=health, tier=tier, golden=golden,
                divine=divine, reborn=reborn, taunt=taunt, tribes=tribes)


def minion_from_population(raw: Dict, names: Dict[str, str],
                           byname: Dict[str, CardKnowledge]) -> Optional[Dict]:
    """Normalize a comp-data board minion (cardID + tags) -> common shape."""
    name = names.get(raw.get("cardID") or raw.get("cardId"), raw.get("cardID"))
    if not name:
        return None
    t = raw.get("tags", {}) or {}
    ck = byname.get(name)
    tribes = ck.tribes if ck else []
    return _minion(name, _i(t.get("ATK")), _i(t.get("HEALTH")),
                   _i(t.get("TECH_LEVEL")) or (ck.tier if ck else 1),
                   _i(t.get("PREMIUM")) == 1, _i(t.get("DIVINE_SHIELD")) == 1,
                   _i(t.get("REBORN")) == 1, _i(t.get("TAUNT")) == 1, tribes)


def minion_from_snapshot(raw: Dict, byname: Dict[str, CardKnowledge]
                         ) -> Optional[Dict]:
    """Normalize a live MinionView dict (name/attack/health/tags) -> common shape."""
    name = raw.get("name") or raw.get("card_id")
    if not name:
        return None
    t = raw.get("tags", {}) or {}
    ck = byname.get(name)
    tribes = ck.tribes if ck else []
    atk = raw.get("attack")
    hp = raw.get("health")
    return _minion(name, _i(atk) if atk is not None else _i(t.get("ATK")),
                   _i(hp) if hp is not None else _i(t.get("HEALTH")),
                   _i(t.get("TECH_LEVEL")) or (ck.tier if ck else 1),
                   _i(t.get("PREMIUM")) == 1, _i(t.get("DIVINE_SHIELD")) == 1,
                   _i(t.get("REBORN")) == 1, _i(t.get("TAUNT")) == 1, tribes)


def emb_dim(emb: Dict[str, List[float]]) -> int:
    return len(next(iter(emb.values()))) if emb else 0


def feature_dim(emb: Dict[str, List[float]]) -> int:
    return emb_dim(emb) + len(TRIBES) + len(_SCALARS)


def board_vector(minions: List[Dict], emb: Dict[str, List[float]]) -> np.ndarray:
    """Dense board features. `minions` are normalized dicts; `emb` is card2vec."""
    dim = emb_dim(emb)
    vecs = [emb[m["name"]] for m in minions if m["name"] in emb]
    c2v = np.mean(np.asarray(vecs, dtype=float), axis=0) if vecs else np.zeros(dim)

    hist = np.zeros(len(TRIBES))
    for m in minions:
        for tr in m["tribes"]:
            if tr in _TRIBE_IX:
                hist[_TRIBE_IX[tr]] += 1
    n = max(len(minions), 1)
    hist /= n

    atks = [m["atk"] for m in minions] or [0]
    hps = [m["health"] for m in minions] or [0]
    tiers = [m["tier"] for m in minions] or [0]
    scalars = np.array([
        len(minions), sum(atks), sum(hps), float(np.mean(atks)),
        float(np.mean(hps)), max(tiers), float(np.mean(tiers)),
        sum(m["golden"] for m in minions), sum(m["divine"] for m in minions),
        sum(m["reborn"] for m in minions), sum(m["taunt"] for m in minions),
    ], dtype=float)
    return np.concatenate([c2v, hist, scalars])
