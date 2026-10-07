"""The ONE state/option encoder for the behaviour-cloned NEXT policy.

Shared by training (ml/train_bc_policy.py) and the live overlay
(hsbg_coach/live.py). HSBG_NEXT_POLICY defaults on; set it to 0, false,
or off to use the eval-net advisor. There is no second copy: if a
feature changes, bump ENCODER_VERSION. Checkpoints record it, and a mismatched
checkpoint refuses to load.

Input is a bg.Snapshot or its dict form (Snapshot.to_dict(), dataclasses.asdict
or a JSON row); every form encodes to the identical vector. Optional State
Builder fields (hero_armor, shop_frozen) are read defensively and default to
absent. anomaly and hand_spells are ignored on purpose: State Builder rows have
anomaly=None and keep spells in `hand`.

Canonical option dict (the Labeler emits exactly this shape)::

    {"type": "buy|sell|reroll|freeze|level|hero_power|play|discover|end_turn|reposition",
     "card_id": str | None,
     "source": {"zone": "board|shop|hand|none", "slot": int | None},
     "target": {"zone": "board|shop|hand|none", "slot": int | None},
     "position": int | None,      # board index for play / reposition
     "choice_kind": str | None}

plus optional debug fields src_entity / target_entity, which are ignored here
(the live overlay has no entity ids for its options, so encoding them would
skew train vs live; the trainer merges options whose vectors coincide, e.g. a
spell aimed at our hero vs at Bob). Slots are 0-based indexes into the snapshot
lists (board / shop / hand), not ZONE_POSITION. Shop spells take the shop slots
after the minions: len(shop) + index into shop_spells. A `play` from zone board
is an Activate; a `play` with source none is the Dark Gift button; a `play`
whose card_id differs from the hand card at its source slot is a Choose One
variant (e.g. BG31_880 -> BG31_880t / BG31_880t2).

Training and held-out scoring use each label row's own `options` list (the
server's legal options); legal_options() below is only the live overlay's
candidate generator. scripts/legal_options_parity.py measures how far apart the
two are.

bc-enc-v2 (from v1): option vector gains [choose-one variant flag, variant
index] so Choose One variants no longer collapse onto one vector; option_cost
knows Activate / Dark Gift costs.

bc-enc-v3 (from v2): reads the cost fields State Builder v1.1 / BGTracker fill
from entity tags (MinionView.buy_cost, shop_spells[].buy_cost, reroll_cost,
hero_power {card_id, cost, usable}, dark_gift {cost, usable}); none of them
comes from the server Options block, so live and training see the same values.
  state  + reroll cost / free / affordable, hero power present / affordable /
           used this turn (not usable although affordable), a 16-bucket hash of
           the hero power card id (which power it is), Dark Gift present /
           usable / cost / affordable, gold minus level cost, cheapest shop buy
           and how many shop buys are affordable, shop tier composition (max /
           mean tier vs tavern tier, share at the current tier), shop minions
           that pair / triple a card we hold, shop minions sharing our main
           tribe, rolls / buys so far this turn.
  option + affordable (gold >= cost); buy: slot buy cost, pair / triple maker,
           shares our main tribe, tier vs tavern tier; shop context for
           freeze / reroll (frozen now, i.e. this unfreezes; triple makers and
           tribe matches in shop; best shop tier vs tavern tier); position
           block for reposition / play (move delta, to leftmost / rightmost,
           board size, the minion's attack / health rank on our board);
           keywords deathrattle / avenge / windfury / venomous / magnetic of
           the acted-on card; 16-bucket hash of a Choose One variant's card id.
option_cost reads the slot's buy_cost for buys (else 3 / the spell's cost), and
legal_options uses buy_cost / reroll_cost for affordability when present.

bc-enc-v4 (from v3): hero-power identity and turn progress.
  state  - the 16-bucket CRC32 hash of the hero power id is replaced by a one-hot
           over HERO_POWER_VOCAB (every hero power id seen in the v2.1 training
           games, + one "other" slot): the hash put 71 ids into 15 buckets and
           the two most common powers (BG36_HERO_000p, BG36_HERO_002p; 43% of
           hero-power picks) each shared a bucket with four other powers.
         + hp_passive: the power is in PASSIVE_HERO_POWERS (the tracker reports
           those as usable, but the server never offers them); hp_usable /
           hp_affordable / hp_used are 0 for them.
         + gold spent this turn (turn-start gold min(turn + 2, 10) minus gold,
           floored at 0) and gold left as a share of turn-start gold.
         The last two v3 scalars are renamed rolls_affordable / buys_affordable:
         they always were gold // cost, not counts of rolls / buys this turn.
  option + the hero power one-hot (same vocabulary) on hero_power options, so
           the scorer sees WHICH power it is scoring (v3's option vector for a
           hero power was identical for every hero); spends the last gold
           (cost > 0 and gold == cost); still able to buy afterwards
           (gold - cost >= the cheapest shop buy).
legal_options no longer offers a hero power listed in PASSIVE_HERO_POWERS.

bc-enc-v5 (from v4): which hero it is, and the hero powers the v2.1 list missed.
  HERO_POWER_VOCAB appends five ids after the frozen v2.1 prefix (indices 0-71
  unchanged; "other" stays last): Sylvanas Windrunner BG23_HERO_306p, C'Thun
  TB_BaconShop_HP_104, Edwin VanCleef TB_BaconShop_HP_001, and the passive
  countdown powers of Morchie (BG34_HERO_004p) and Murozond (BG34_HERO_000p).
  Those two countdowns join PASSIVE_HERO_POWERS: the tracker reports them
  usable, but the server never offers a click. HP_ALIASES (empty) lets a future
  skin or upgraded '...p2' id share an existing slot without a dimension change.
  state  + one-hot of canonical_hero_id(state.hero) over HERO_VOCAB (the 117
           base heroes seen in the v2 / 36.6.3 corpora, most frequent first,
           + one "other" slot). A trailing _SKIN_* is stripped, then
           HERO_SKIN_ALIASES maps reskinned shop ids onto the base hero
           (Moonwell Sylvanas TB_BaconShop_HERO_44_SKIN_D -> BG23_HERO_306).
           The hero-select placeholder TB_BaconShop_HERO_PH is no hero: all
           zeros, not the "other" slot.
  option + the same hero one-hot when choice_kind == "hero", zeros otherwise,
           so a hero pick encodes which hero it is. Before this, those options
           differed only by source slot.

Needs numpy (the ml extra). The stdlib-only core never imports this module
unless the policy flag is on.
"""

import re
import zlib
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from .actions import BUY_COST, MAX_BOARD, MAX_TIER, ROLL_COST, SELL_VALUE, UPGRADE_COST

ENCODER_VERSION = "bc-enc-v5"

MAX_HAND = 10
EMB_DIM = 48                            # card2vec width (data/cards/card2vec.json)
OPTION_TYPES = ("buy", "sell", "reroll", "freeze", "level", "hero_power",
                "play", "discover", "end_turn", "reposition")
ZONES = ("board", "shop", "hand", "none")
OPTION_KEYS = ("type", "card_id", "source", "target", "position", "choice_kind")

_N_TRIBES = 11                          # len(ml.board_features.TRIBES)
_ZONE_DIM = EMB_DIM + _N_TRIBES + 11    # board_vector: card2vec | tribes | 11 scalars
# Fixed scale for board_vector's scalars (size, sum atk/hp, mean atk/hp,
# max/mean tier, golden, divine, reborn, taunt) so the MLP sees ~unit inputs.
_ZONE_SCALE = np.array([7, 50, 50, 10, 10, 6, 6, 7, 7, 7, 7], dtype=np.float64)
STATE_SCALARS = ("turn", "tier", "gold", "health", "armor", "n_board", "n_shop",
                 "n_hand", "level_cost", "can_level", "hp_usable", "hp_cost",
                 "n_shop_spells", "shop_frozen", "n_hand_spells",
                 # bc-enc-v3
                 "reroll_cost", "reroll_free", "can_reroll", "hp_present",
                 "hp_affordable", "hp_used", "dg_present", "dg_usable", "dg_cost",
                 "dg_affordable", "gold_minus_level", "min_buy_cost", "n_buy_affordable",
                 "shop_max_tier_rel", "shop_mean_tier_rel", "shop_at_tier",
                 "shop_pairs", "shop_triples", "shop_tribe_match", "rolls_affordable",
                 "buys_affordable",
                 # bc-enc-v4
                 "hp_passive", "gold_spent_turn", "gold_left_share")
HASH_BUCKETS = 16
# Hero power card ids, most frequent first: the v2.1 training games (build
# 253216, held-out games excluded) plus the 36.6.3 v3.12 additions appended
# at the end. Ids not listed share the trailing "other" slot. Append new ids
# (and bump ENCODER_VERSION); never reorder existing ones.
HERO_POWER_VOCAB = (
    'BG36_HERO_000p', 'BG36_HERO_002p', 'BG21_HERO_000p', 'TB_BaconShop_HP_020',
    'TB_BaconShop_HP_010', 'TB_BaconShop_HP_024', 'BG25_HERO_103p',
    'TB_BaconShop_HP_046', 'TB_BaconShop_HP_075', 'BG25_HERO_105p', 'BG26_HERO_102p',
    'BG20_HERO_301p', 'BG21_HERO_010p', 'BG34_HERO_001p', 'BG28_HERO_801p',
    'BG31_HERO_003p', 'BG31_HERO_005p', 'BG20_HERO_101p', 'BG20_HERO_201p',
    'TB_BaconShop_HP_011', 'TB_BaconShop_HP_103', 'BG26_HERO_102p2', 'BG26_HERO_104p',
    'TB_BaconShop_HP_053', 'BG32_HERO_001p', 'TB_BaconShop_HP_047', 'BG21_HERO_020p',
    'TB_BaconShop_HP_077', 'TB_BaconShop_HP_081', 'TB_BaconShop_HP_072',
    'TB_BaconShop_HP_022', 'BG22_HERO_000p_Alt', 'BG23_HERO_303p2', 'BG31_HERO_006p',
    'TB_BaconShop_HP_052', 'TB_BaconShop_HP_042', 'TB_BaconShop_HP_028',
    'BG22_HERO_001p', 'TB_BaconShop_HP_074', 'TB_BaconShop_HP_084', 'BG20_HERO_280p5',
    'BG23_HERO_305p', 'TB_BaconShop_HP_702t', 'TB_BaconShop_HP_068',
    'TB_BaconShop_HP_040', 'TB_BaconShop_HP_041k', 'TB_BaconShop_HP_064',
    'TB_BaconShop_HP_102', 'TB_BaconShop_HP_076', 'TB_BaconShop_HP_041l',
    'BG24_HERO_100p', 'BG20_HERO_282p', 'BG20_HERO_103p', 'BG28_HERO_400p',
    'TB_BaconShop_HP_038', 'TB_BaconShop_HP_041d', 'BG31_HERO_811p2',
    'TB_BaconShop_HP_041b', 'TB_BaconShop_HP_049', 'BG20_HERO_283p',
    'TB_BaconShop_HP_041i', 'TB_BaconShop_HP_041h', 'TB_BaconShop_HP_041a',
    'TB_BaconShop_HP_041g', 'TB_BaconShop_HP_036', 'BG31_HERO_811p',
    'TB_BaconShop_HP_041c', 'BG20_HERO_201p2', 'TB_BaconShop_HP_015',
    'TB_BaconShop_HP_041f', 'BG25_HERO_100p', 'TB_BaconShop_HP_057',
    # bc-enc-v5: 36.6.3 v3.12 additions (indices 72-76). Do not insert above.
    'BG23_HERO_306p',       # Sylvanas Windrunner, Reclaimed Souls (active)
    'TB_BaconShop_HP_104',  # C'Thun, Saturday C'Thuns! (active)
    'TB_BaconShop_HP_001',  # Edwin VanCleef, Sharpen Blades (active)
    'BG34_HERO_004p',       # Morchie, Warped Conflux (passive countdown)
    'BG34_HERO_000p',       # Murozond, Unbounded, Alternate Timeline (passive countdown)
)
HP_VOCAB_DIM = len(HERO_POWER_VOCAB) + 1
_HP_INDEX = {cid: i for i, cid in enumerate(HERO_POWER_VOCAB)}
# Skin or upgraded power ids that share a vocab slot. Empty until one shows up
# in the labels; lookup happens in canonical_hp_id, so adding a mapping does
# not change HP_VOCAB_DIM.
HP_ALIASES: Dict[str, str] = {}
# Hero powers the tracker reports as usable although they cannot be clicked.
# v2.1 labels: the server never listed them as an option in any of their
# (usable, affordable) decision points (TB_BaconShop_HP_042 2131 rows,
# BG20_HERO_280p5 1679, TB_BaconShop_HP_038 1299, BG24_HERO_100p 984,
# BG20_HERO_282p 803, TB_BaconShop_HP_036 236, TB_BaconShop_HP_057 13).
# 36.6.3: Morchie / Murozond countdown text ("On Turn N, visit the Minor/Major
# Timewarp") is the same shape — reported usable, never offered.
PASSIVE_HERO_POWERS = frozenset((
    "TB_BaconShop_HP_042", "BG20_HERO_280p5", "TB_BaconShop_HP_038", "BG24_HERO_100p",
    "BG20_HERO_282p", "TB_BaconShop_HP_036", "TB_BaconShop_HP_057",
    "BG34_HERO_004p", "BG34_HERO_000p"))
# Base hero card ids seen in the v2 / 36.6.3 corpora, most frequent first.
# Skins canonicalize onto these (canonical_hero_id). Ids not listed share the
# trailing "other" slot. Append new ids (and bump ENCODER_VERSION); never reorder.
HERO_VOCAB = (
    'BG36_HERO_002', 'BG36_HERO_000', 'BG36_HERO_105', 'BG21_HERO_000',
    'TB_BaconShop_HERO_74', 'BG35_HERO_001', 'BG26_HERO_102', 'TB_BaconShop_HERO_59',
    'BG20_HERO_202', 'BG24_HERO_100', 'BG20_HERO_242', 'BG33_HERO_001',
    'TB_BaconShop_HERO_21', 'BG21_HERO_020', 'BG30_HERO_304', 'BG28_HERO_400',
    'BG20_HERO_283', 'TB_BaconShop_HERO_67', 'BG23_HERO_303', 'BG26_HERO_104',
    'TB_BaconShop_HERO_27', 'TB_BaconShop_HERO_35', 'TB_BaconShop_HERO_75', 'TB_BaconShop_HERO_39',
    'BG23_HERO_305', 'TB_BaconShop_HERO_36', 'TB_BaconShop_HERO_76', 'TB_BaconShop_HERO_94',
    'BG22_HERO_003', 'TB_BaconShop_HERO_14', 'TB_BaconShop_HERO_42', 'BG31_HERO_003',
    'TB_BaconShop_HERO_45', 'BG22_HERO_004', 'TB_BaconShop_HERO_38', 'BG28_HERO_801',
    'BG31_HERO_811', 'BG20_HERO_201', 'BG22_HERO_002', 'BG36_HERO_101',
    'BG25_HERO_105', 'TB_BaconShop_HERO_12', 'TB_BaconShop_HERO_40', 'TB_BaconShop_HERO_57',
    'BG24_HERO_204', 'BG20_HERO_280', 'BG23_HERO_201', 'BG22_HERO_001',
    'BG32_HERO_001', 'TB_BaconShop_HERO_58', 'TB_BaconShop_HERO_72', 'TB_BaconShop_HERO_60',
    'TB_BaconShop_HERO_41', 'TB_BaconShop_HERO_52', 'BG27_HERO_801', 'TB_BaconShop_HERO_11',
    'BG31_HERO_006', 'TB_BaconShop_HERO_15', 'BG20_HERO_301', 'TB_BaconShop_HERO_49',
    'TB_BaconShop_HERO_90', 'TB_BaconShop_HERO_28', 'BG34_HERO_001', 'TB_BaconShop_HERO_34',
    'BG20_HERO_101', 'TB_BaconShop_HERO_64', 'BG34_HERO_002', 'BG31_HERO_005',
    'TB_BaconShop_HERO_01', 'BG22_HERO_201', 'BG32_HERO_002', 'TB_BaconShop_HERO_02',
    'BG21_HERO_010', 'BG22_HERO_000', 'TB_BaconShop_HERO_50', 'BG22_HERO_200',
    'BG28_HERO_800', 'BG20_HERO_100', 'TB_BaconShop_HERO_43', 'TB_BaconShop_HERO_68',
    'TB_BaconShop_HERO_92', 'BG20_HERO_102', 'TB_BaconShop_HERO_10', 'TB_BaconShop_HERO_23',
    'TB_BaconShop_HERO_70', 'TB_BaconShop_HERO_62', 'BG26_HERO_101', 'BG22_HERO_305',
    'TB_BaconShop_HERO_22', 'BG21_HERO_030', 'TB_BaconShop_HERO_16', 'TB_BaconShop_HERO_29',
    'TB_BaconShop_HERO_18', 'BG20_HERO_103', 'TB_BaconShop_HERO_08', 'TB_BaconShop_HERO_71',
    'TB_BaconShop_HERO_55', 'TB_BaconShop_HERO_53', 'BG31_HERO_801', 'BG25_HERO_100',
    'TB_BaconShop_HERO_33', 'TB_BaconShop_HERO_702', 'BG31_HERO_802', 'TB_BaconShop_HERO_56',
    'TB_BaconShop_HERO_78', 'TB_BaconShop_HERO_95', 'BG25_HERO_103', 'TB_BaconShop_HERO_25',
    'TB_BaconShop_HERO_91', 'BG20_HERO_282', 'TB_BaconShop_HERO_37', 'TB_BaconShop_HERO_93',
    'TB_BaconShop_HERO_17', 'BG23_HERO_306', 'BG34_HERO_004', 'BG34_HERO_000',
    'TB_BaconShop_HERO_59t',
)
HERO_VOCAB_DIM = len(HERO_VOCAB) + 1
_HERO_INDEX = {cid: i for i, cid in enumerate(HERO_VOCAB)}
# Shop card id -> base hero, from HearthstoneJSON heroPowerDbfId /
# BACON_SKIN_PARENT_ID. Applied after the _SKIN_* suffix is stripped, so
# TB_BaconShop_HERO_44_SKIN_D (Moonwell Sylvanas, parent dbf 89293) is
# BG23_HERO_306.
HERO_SKIN_ALIASES = {
    'TB_BaconShop_HERO_44': 'BG23_HERO_306',
    'TB_BaconShop_HERO_102': 'BG20_HERO_102',
    'TB_BaconShop_HERO_201': 'BG20_HERO_201',
    'TB_BaconShop_HERO_103': 'BG20_HERO_103',
}
# Hero-select placeholder. Not a hero: the one-hot stays zeros, not "other".
_NO_HERO = frozenset(("TB_BaconShop_HERO_PH",))
_SKIN_SUFFIX = re.compile(r"_SKIN_[A-Z0-9]+$")
STATE_DIM = (3 * _ZONE_DIM + len(STATE_SCALARS) + HP_VOCAB_DIM
             + HERO_VOCAB_DIM)  # hero power one-hot, then hero one-hot
_CARD_DIM = EMB_DIM + 7 + _N_TRIBES + 1              # card2vec | stats | tribes | spell
KEYWORDS = ("DEATHRATTLE", "AVENGE", "WINDFURY", "VENOMOUS", "MAGNETIC")
_V3_OPTION = 1 + 5 + 5 + 6 + len(KEYWORDS) + HASH_BUCKETS
_V4_OPTION = HP_VOCAB_DIM + 2
_V5_OPTION = HERO_VOCAB_DIM                 # hero one-hot on hero-pick options
OPTION_DIM = (len(OPTION_TYPES) + _CARD_DIM + 2 * (len(ZONES) + 1) + 2 + 2 + 1 + 2
              + _V3_OPTION + _V4_OPTION + _V5_OPTION)

_SPELL_TYPES = ("BATTLEGROUND_SPELL", "SPELL")
_NOT_A_CARD = ("ENCHANTMENT", "HERO", "HERO_POWER", "GAME_MODE_BUTTON")


# --- small readers (Snapshot object or dict) --------------------------------
def _get(obj, key, default=None):
    v = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return default if v is None else v


def _int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_dict(m) -> Dict:
    if isinstance(m, dict):
        return m
    return {"entity_id": getattr(m, "entity_id", None),
            "card_id": getattr(m, "card_id", None), "name": getattr(m, "name", None),
            "attack": getattr(m, "attack", None), "health": getattr(m, "health", None),
            "position": getattr(m, "position", None),
            "buy_cost": getattr(m, "buy_cost", None),
            "tags": getattr(m, "tags", None) or {}}


def _tags(card) -> Dict:
    return (card.get("tags") or {}) if isinstance(card, dict) else {}


def _zone(snapshot, key) -> List[Dict]:
    return [_as_dict(m) for m in (_get(snapshot, key) or [])]


@lru_cache(maxsize=1)
def _resources():
    """(card2vec by name, CardKnowledge by name, CardKnowledge by card id)."""
    from hsbg_coach.cards import by_name, load_kb
    from hsbg_coach.synergy import load_embeddings
    emb = {k: v for k, v in load_embeddings().items() if len(v) == EMB_DIM}
    kb = load_kb()
    return emb, by_name(kb), kb


def is_spell(card) -> bool:
    return _tags(card).get("CARDTYPE") in _SPELL_TYPES


def needs_board_slot(card) -> bool:
    """Minions (CARDTYPE MINION or untagged) need a free board slot to play."""
    return _tags(card).get("CARDTYPE") in (None, "", "MINION")


def playable_hand(snapshot) -> List[tuple]:
    """(hand index, card) for real cards in hand; skips DragBuy/placeholder chrome."""
    out = []
    for i, m in enumerate(_zone(snapshot, "hand")):
        cid = m.get("card_id") or ""
        if not cid or "DragBuy" in cid or "UNKNOWN ENTITY" in (m.get("name") or ""):
            continue
        if _tags(m).get("CARDTYPE") in _NOT_A_CARD:
            continue
        out.append((i, m))
    return out


def level_cost(snapshot) -> Optional[int]:
    """Gold to tier up now: the live discounted cost, else the base cost."""
    tier = _int(_get(snapshot, "tavern_tier"), 1) or 1
    if tier >= MAX_TIER:
        return None
    cost = _get(snapshot, "level_cost")
    return _int(cost) if cost is not None else UPGRADE_COST.get(tier)


def _spell_cost(spell) -> int:
    c = _get(spell, "cost")
    return BUY_COST if c is None else _int(c, BUY_COST)


def _hero_power(snapshot) -> Optional[Dict]:
    hp = _get(snapshot, "hero_power")
    return hp if isinstance(hp, dict) else None


def canonical_hp_id(card_id) -> Optional[str]:
    """Hero power id after HP_ALIASES. None when there is no power."""
    if not card_id:
        return None
    cid = str(card_id)
    return HP_ALIASES.get(cid, cid)


def canonical_hero_id(card_id) -> Optional[str]:
    """Base hero card id: strip a trailing ``_SKIN_*``, then HERO_SKIN_ALIASES.

    None, empty, and the hero-select placeholder ``TB_BaconShop_HERO_PH`` are
    no hero (the one-hot is all zeros, not the "other" slot). Any other id is
    returned even when it is not in HERO_VOCAB; hero_onehot puts those in
    "other".
    """
    if card_id is None:
        return None
    cid = str(card_id).strip()
    if not cid or cid in _NO_HERO:
        return None
    cid = _SKIN_SUFFIX.sub("", cid)
    if not cid or cid in _NO_HERO:
        return None
    return HERO_SKIN_ALIASES.get(cid, cid)


def is_passive_hero_power(hp: Optional[Dict]) -> bool:
    cid = hp.get("card_id") if isinstance(hp, dict) else None
    return canonical_hp_id(cid) in PASSIVE_HERO_POWERS


def hero_power_onehot(card_id) -> List[float]:
    """One-hot over HERO_POWER_VOCAB + "other" (all zeros for no hero power).

    HP_ALIASES ids share the target's slot.
    """
    out = [0.0] * HP_VOCAB_DIM
    cid = canonical_hp_id(card_id)
    if cid:
        out[_HP_INDEX.get(cid, HP_VOCAB_DIM - 1)] = 1.0
    return out


def hero_onehot(card_id) -> List[float]:
    """One-hot over HERO_VOCAB + "other".

    All zeros when canonical_hero_id returns None (no hero, or the hero-select
    placeholder). An id that is not in the vocabulary takes the last slot.
    """
    out = [0.0] * HERO_VOCAB_DIM
    cid = canonical_hero_id(card_id)
    if cid:
        out[_HERO_INDEX.get(cid, HERO_VOCAB_DIM - 1)] = 1.0
    return out


def turn_start_gold(snapshot) -> int:
    """Base gold at the start of this turn: 3 on turn 1, +1 per turn, max 10."""
    return min(max(_int(_get(snapshot, "turn"), 1), 1) + 2, 10)


def _dark_gift(snapshot) -> Optional[Dict]:
    dg = _get(snapshot, "dark_gift")
    return dg if isinstance(dg, dict) and dg.get("card_id") else None


def id_hash(card_id, buckets: int = HASH_BUCKETS) -> List[float]:
    """One-hot of a stable CRC32 bucket of a card id (all zeros for none).
    CRC32, not hash(): Python's str hash is salted per process."""
    out = [0.0] * buckets
    if card_id:
        out[zlib.crc32(str(card_id).encode("utf-8")) % buckets] = 1.0
    return out


def reroll_cost(snapshot) -> int:
    """Gold to refresh now: snapshot.reroll_cost (the button's COST, 0 = free)
    when the tracker / State Builder filled it, else the base cost."""
    rc = _get(snapshot, "reroll_cost")
    return ROLL_COST if rc is None else _int(rc, ROLL_COST)


def buy_cost(snapshot, slot: int) -> int:
    """Gold to buy shop slot `slot` (minions first, then shop_spells): the slot's
    buy handle COST (MinionView.buy_cost / shop_spells[].buy_cost) when filled,
    else 3 for a minion / the spell's own cost."""
    shop = _zone(snapshot, "shop")
    if 0 <= slot < len(shop):
        bc = shop[slot].get("buy_cost")
        return BUY_COST if bc is None else _int(bc, BUY_COST)
    spells = _get(snapshot, "shop_spells") or []
    k = slot - len(shop)
    if 0 <= k < len(spells):
        bc = _get(spells[k], "buy_cost")
        return _spell_cost(spells[k]) if bc is None else _int(bc, BUY_COST)
    return BUY_COST


def is_unplayable(card) -> bool:
    """The game flags the card as unplayable right now (LITERALLY_UNPLAYABLE=1,
    e.g. a Lockbox or a spell whose condition isn't met)."""
    return str(_tags(card).get("LITERALLY_UNPLAYABLE", "")) == "1"


def activation(snapshot, slot: int, minion: Dict) -> Optional[Dict]:
    """{cost, usable} if the board minion at `slot` has an Activate ability.

    The raw tag INTERACTABLE_OBJECT, when present, is the game's own signal:
    1 = clickable now; legal if gold >= INTERACTABLE_OBJECT_COST (absent = 0).
    Without that tag, the minion must be listed in snapshot.activatable (by
    entity id, else card id; the tracker / State Builder fill it from the
    Activate allowlist in hsbg_coach/activate_cards.py) and its entry's `usable`
    flag decides. Pilot check (5 games, targets ignored): 499 server-listed
    Activates, 2 missed, 63 extra (scripts/legal_options_parity.py)."""
    tags = _tags(minion)
    entries = [a for a in (_get(snapshot, "activatable") or []) if isinstance(a, dict)]
    eid, cid = minion.get("entity_id"), minion.get("card_id")
    entry = next((a for a in entries if eid is not None and a.get("entity_id") == eid), None)
    if entry is None:
        entry = next((a for a in entries if cid and a.get("card_id") == cid
                      and a.get("entity_id") is None), None)
    io = tags.get("INTERACTABLE_OBJECT")
    if io is not None:
        if str(io) != "1" and entry is None:
            return None
        raw = tags.get("INTERACTABLE_OBJECT_COST")
        need = _int(raw, 0) if raw is not None else 0
        cost = need if raw is not None or entry is None else _int(entry.get("cost"), 0)
        gold = _int(_get(snapshot, "gold"), 0)
        return {"cost": cost, "usable": str(io) == "1" and gold >= need}
    if entry is None:
        return None
    return {"cost": _int(entry.get("cost"), 0), "usable": bool(entry.get("usable"))}


# --- shared context (bc-enc-v3) ---------------------------------------------
_CTX_CACHE: list = [None, None]          # [snapshot object, context]: 1-entry cache


def _minion_info(card: Dict) -> Optional[Dict]:
    from ml.board_features import minion_from_snapshot
    return minion_from_snapshot(card, _resources()[1])


def _golden(card: Dict) -> bool:
    return str(_tags(card).get("PREMIUM", "")) == "1"


def _context(snapshot) -> Dict:
    """Per-snapshot numbers shared by encode_state and encode_option. Cached
    for the last snapshot object seen (a strong reference, so its id cannot be
    reused); snapshots are not mutated while their options are encoded."""
    if _CTX_CACHE[0] is snapshot:
        return _CTX_CACHE[1]
    board, shop = _zone(snapshot, "board"), _zone(snapshot, "shop")
    hand = [m for _, m in playable_hand(snapshot)]
    spells = _get(snapshot, "shop_spells") or []
    gold = _int(_get(snapshot, "gold"), 0)
    tier = _int(_get(snapshot, "tavern_tier"), 1) or 1
    held: Dict[str, int] = {}
    for m in board + hand:
        cid = m.get("card_id")
        if cid and not _golden(m) and needs_board_slot(m):
            held[cid] = held.get(cid, 0) + 1
    counts: Dict[str, int] = {}
    board_info = [_minion_info(m) for m in board]
    for mi in board_info:
        for tr in (mi or {}).get("tribes") or []:
            if tr != "All":
                counts[tr] = counts.get(tr, 0) + 1
    main_tribe = None
    if counts:
        from ml.board_features import TRIBES
        order = {t: i for i, t in enumerate(TRIBES)}
        main_tribe = min(counts, key=lambda t: (-counts[t], order.get(t, 99), t))
    shop_info = [_minion_info(m) for m in shop]
    tiers = [(mi or {}).get("tier") or 1 for mi in shop_info]
    pair = [bool(m.get("card_id")) and held.get(m.get("card_id"), 0) >= 1 for m in shop]
    triple = [bool(m.get("card_id")) and held.get(m.get("card_id"), 0) >= 2 for m in shop]
    match = [main_tribe is not None and bool(set((mi or {}).get("tribes") or [])
                                             & {main_tribe, "All"}) for mi in shop_info]
    costs = [buy_cost(snapshot, i) for i in range(len(shop) + len(spells))]
    atks = [(mi or {}).get("atk", 0) for mi in board_info]
    hps = [(mi or {}).get("health", 0) for mi in board_info]
    ctx = {"board": board, "shop": shop, "hand": hand, "gold": gold, "tier": tier,
           "held": held, "main_tribe": main_tribe, "shop_info": shop_info,
           "tiers": tiers, "pair": pair, "triple": triple, "match": match,
           "costs": costs, "atks": atks, "hps": hps}
    _CTX_CACHE[0], _CTX_CACHE[1] = snapshot, ctx
    return ctx


# --- state ------------------------------------------------------------------
def _zone_block(cards: List[Dict]) -> np.ndarray:
    from ml.board_features import board_vector, minion_from_snapshot
    emb, byname, _ = _resources()
    minions = [x for x in (minion_from_snapshot(c, byname) for c in cards) if x]
    v = board_vector(minions, emb or {"": [0.0] * EMB_DIM})
    v[-11:] = v[-11:] / _ZONE_SCALE
    return v


def encode_state(snapshot) -> np.ndarray:
    """Fixed STATE_DIM float32 vector: board | shop | hand + scalars
    + hero-power one-hot + hero one-hot."""
    board, shop = _zone(snapshot, "board"), _zone(snapshot, "shop")
    hand = [m for _, m in playable_hand(snapshot)]
    gold = _int(_get(snapshot, "gold"), 0)
    lc = level_cost(snapshot)
    hp = _hero_power(snapshot) or {}
    passive = is_passive_hero_power(hp)
    scalars = np.array([
        _int(_get(snapshot, "turn"), 0) / 20.0,
        _int(_get(snapshot, "tavern_tier"), 1) / 6.0,
        gold / 10.0,
        _int(_get(snapshot, "hero_health"), 0) / 40.0,
        _int(_get(snapshot, "hero_armor"), 0) / 20.0,
        len(board) / MAX_BOARD,
        len(shop) / MAX_BOARD,
        len(hand) / MAX_HAND,
        (lc or 0) / 10.0,
        1.0 if lc is not None and gold >= lc else 0.0,
        1.0 if hp.get("usable") and not passive else 0.0,
        _int(hp.get("cost"), 0) / 10.0,
        len(_get(snapshot, "shop_spells") or []) / 3.0,
        1.0 if _get(snapshot, "shop_frozen") else 0.0,
        sum(1 for m in hand if is_spell(m)) / MAX_HAND,
    ], dtype=np.float64)
    ctx = _context(snapshot)
    tier = ctx["tier"]
    rc = reroll_cost(snapshot)
    hp_cost = _int(hp.get("cost"), 0)
    dg = _dark_gift(snapshot) or {}
    dg_cost = _int(dg.get("cost"), 0)
    costs, tiers, n_shop = ctx["costs"], ctx["tiers"], max(len(shop), 1)
    min_buy = min(costs) if costs else BUY_COST
    v3 = np.array([
        rc / 5.0,
        1.0 if rc == 0 else 0.0,
        1.0 if gold >= rc else 0.0,
        1.0 if hp else 0.0,
        1.0 if hp and not passive and gold >= hp_cost else 0.0,
        1.0 if hp and not passive and not hp.get("usable") and gold >= hp_cost else 0.0,
        1.0 if dg else 0.0,
        1.0 if dg.get("usable") else 0.0,
        dg_cost / 10.0,
        1.0 if dg and gold >= dg_cost else 0.0,
        (gold - lc) / 10.0 if lc is not None else 0.0,
        min_buy / 3.0,
        sum(1 for c in costs if gold >= c) / MAX_BOARD,
        (max(tiers) - tier) / 6.0 if tiers else 0.0,
        (sum(tiers) / len(tiers) - tier) / 6.0 if tiers else 0.0,
        sum(1 for t in tiers if t == tier) / n_shop if tiers else 0.0,
        sum(ctx["pair"]) / MAX_BOARD,
        sum(ctx["triple"]) / MAX_BOARD,
        sum(ctx["match"]) / MAX_BOARD,
        min(gold // rc, 10) / 10.0 if rc > 0 else 1.0,
        min(gold // min_buy, 10) / 10.0 if min_buy > 0 else 1.0,
    ], dtype=np.float64)
    start = turn_start_gold(snapshot)
    v4 = np.array([
        1.0 if passive else 0.0,
        max(start - gold, 0) / 10.0,
        min(gold / start, 2.0),
    ], dtype=np.float64)
    return np.concatenate([_zone_block(board), _zone_block(shop), _zone_block(hand),
                           scalars, v3, v4, hero_power_onehot(hp.get("card_id")),
                           hero_onehot(_get(snapshot, "hero"))]
                          ).astype(np.float32)


# --- options ----------------------------------------------------------------
def make_option(type_: str, card_id: Optional[str] = None, source=("none", None),
                target=("none", None), position: Optional[int] = None,
                choice_kind: Optional[str] = None) -> Dict:
    return {"type": type_, "card_id": card_id,
            "source": {"zone": source[0], "slot": source[1]},
            "target": {"zone": target[0], "slot": target[1]},
            "position": position, "choice_kind": choice_kind}


def option_key(option: Dict) -> tuple:
    """Hashable identity of an option (debug fields ignored), for matching."""
    src, tgt = option.get("source") or {}, option.get("target") or {}
    return (option.get("type"), option.get("card_id"),
            src.get("zone") or "none", src.get("slot"),
            tgt.get("zone") or "none", tgt.get("slot"),
            option.get("position"), option.get("choice_kind"))


def legal_options(snapshot) -> List[Dict]:
    """Legal actions at a recruit-phase decision point, in a stable order.

    Live-overlay candidate generator only: training and scoring use each label
    row's own server option list. Gold / space rules plus what the Snapshot
    carries (card-specific exceptions are not modelled):
      buy        gold >= the slot's buy cost (buy_cost(): the buy handle's COST
                 when the Snapshot has it, else 3 per minion / the spell's own
                 cost); needs a free hand slot (< 10 cards). A full board does
                 NOT block buys.
      sell       every board minion, always
      play       hand card not flagged LITERALLY_UNPLAYABLE; a spell with a
                 COST tag needs gold >= COST. Minion: board < 7, one option per
                 insertion index 0..len(board); spell / non-minion card: one
                 option, position None. Always target none (targets need card
                 data the repo does not have); a Choose One card is emitted as
                 its parent (variant ids are not in the Snapshot).
      activate   `play` from zone board for each board minion whose activation()
                 is usable; target none
      dark gift  `play` of snapshot.dark_gift's card_id with source none, when
                 it is usable
      reposition board >= 2: every (slot, new index) pair with index != slot
      reroll     gold >= reroll_cost() (snapshot.reroll_cost, 0 = free; else 1)
      freeze     always (it toggles)
      level      tier < 6 and gold >= level_cost (live discounted, else base)
      hero_power hero_power present, usable, not in PASSIVE_HERO_POWERS, and
                 gold >= its cost; no target
      end_turn   always
      discover   never enumerated: a Snapshot carries no choice offer
    """
    if _get(snapshot, "phase") not in (None, "recruit", "unknown"):
        return []
    gold = _int(_get(snapshot, "gold"), 0)
    board, shop = _zone(snapshot, "board"), _zone(snapshot, "shop")
    hand = playable_hand(snapshot)
    opts: List[Dict] = []
    if len(hand) < MAX_HAND:
        for i, m in enumerate(shop):
            if gold >= buy_cost(snapshot, i):
                opts.append(make_option("buy", m.get("card_id"), ("shop", i)))
        for j, sp in enumerate(_get(snapshot, "shop_spells") or []):
            if gold >= buy_cost(snapshot, len(shop) + j):
                opts.append(make_option("buy", _get(sp, "card_id"), ("shop", len(shop) + j)))
    for i, m in enumerate(board):
        opts.append(make_option("sell", m.get("card_id"), ("board", i)))
    for i, m in hand:
        if is_unplayable(m):
            continue
        if needs_board_slot(m):
            if len(board) < MAX_BOARD:
                for p in range(len(board) + 1):
                    opts.append(make_option("play", m.get("card_id"), ("hand", i), position=p))
        else:
            cost = _tags(m).get("COST")
            if is_spell(m) and cost is not None and gold < _int(cost, 0):
                continue
            opts.append(make_option("play", m.get("card_id"), ("hand", i)))
    for i, m in enumerate(board):
        act = activation(snapshot, i, m)
        if act and act["usable"]:
            opts.append(make_option("play", m.get("card_id"), ("board", i)))
    dg = _dark_gift(snapshot)
    if dg and dg.get("usable"):
        opts.append(make_option("play", dg.get("card_id")))
    if len(board) >= 2:
        for i, m in enumerate(board):
            for p in range(len(board)):
                if p != i:
                    opts.append(make_option("reposition", m.get("card_id"), ("board", i),
                                            position=p))
    if gold >= reroll_cost(snapshot):
        opts.append(make_option("reroll"))
    opts.append(make_option("freeze"))
    lc = level_cost(snapshot)
    if lc is not None and gold >= lc:
        opts.append(make_option("level"))
    hp = _hero_power(snapshot)
    if (hp and hp.get("usable") and not is_passive_hero_power(hp)
            and gold >= _int(hp.get("cost"), 0)):
        opts.append(make_option("hero_power", hp.get("card_id")))
    opts.append(make_option("end_turn"))
    return opts


def option_card(snapshot, option: Dict) -> Optional[Dict]:
    """The card an option acts on: the snapshot entry at source zone/slot, else
    the hero power, else a KB lookup of card_id (e.g. discover picks)."""
    src = option.get("source") or {}
    zone, slot = src.get("zone"), src.get("slot")
    card = None
    if isinstance(slot, int) and slot >= 0 and zone in ("board", "shop", "hand"):
        cards = _zone(snapshot, zone)
        if slot < len(cards):
            card = cards[slot]
        elif zone == "shop":
            spells = _get(snapshot, "shop_spells") or []
            k = slot - len(cards)
            if k < len(spells):
                sp = dict(_as_dict(spells[k]))
                sp["tags"] = dict(sp.get("tags") or {}, CARDTYPE="BATTLEGROUND_SPELL")
                card = sp
    if card is None and option.get("type") == "hero_power":
        hp = _hero_power(snapshot)
        if hp:
            card = {"name": hp.get("name"), "card_id": hp.get("card_id"), "tags": {}}
    if card is None and is_dark_gift(option):
        dg = _dark_gift(snapshot) or {}
        card = {"name": dg.get("name") or "Dark Gift",
                "card_id": option.get("card_id") or dg.get("card_id"), "tags": {}}
    cid = option.get("card_id")
    if card is None and cid:
        ck = _resources()[2].get(cid)
        card = {"card_id": cid, "name": ck.name if ck else cid,
                "attack": ck.attack if ck else None,
                "health": ck.health if ck else None, "tags": {}}
    return card


def is_activate(option: Dict) -> bool:
    return option.get("type") == "play" and (option.get("source") or {}).get("zone") == "board"


def is_dark_gift(option: Dict) -> bool:
    return (option.get("type") == "play"
            and ((option.get("source") or {}).get("zone") or "none") == "none")


def variant_index(snapshot, option: Dict) -> int:
    """0 unless this is a Choose One variant play (card_id differs from the hand
    card at its source slot); then the variant's ordinal from its card-id
    suffix: <parent>t -> 1, <parent>t2 -> 2, <parent>a / b -> 1 / 2, else 1."""
    src = option.get("source") or {}
    cid = option.get("card_id")
    if option.get("type") != "play" or src.get("zone") != "hand" or not cid:
        return 0
    card = option_card(snapshot, option) or {}
    parent = card.get("card_id")
    if not parent or parent == cid:
        return 0
    suffix = cid[len(parent):] if cid.startswith(parent) else cid
    digits = ""
    while suffix and suffix[-1].isdigit():
        digits = suffix[-1] + digits
        suffix = suffix[:-1]
    if digits:
        return max(1, _int(digits, 1))
    if len(suffix) == 1 and suffix.isalpha() and suffix.lower() != "t":
        return max(1, ord(suffix.lower()) - ord("a") + 1)
    return 1


def option_cost(snapshot, option: Dict) -> int:
    """Gold the option spends (negative = gained)."""
    t = option.get("type")
    if is_activate(option):
        slot = (option.get("source") or {}).get("slot")
        board = _zone(snapshot, "board")
        if isinstance(slot, int) and 0 <= slot < len(board):
            act = activation(snapshot, slot, board[slot])
            return act["cost"] if act else 0
        return 0
    if is_dark_gift(option):
        return _int((_dark_gift(snapshot) or {}).get("cost"), 0)
    if t == "buy":
        slot = (option.get("source") or {}).get("slot")
        if isinstance(slot, int) and not isinstance(slot, bool) and slot >= 0:
            return buy_cost(snapshot, slot)
        card = option_card(snapshot, option)
        return _spell_cost(card) if card is not None and is_spell(card) else BUY_COST
    if t == "sell":
        return -SELL_VALUE
    if t == "reroll":
        return reroll_cost(snapshot)
    if t == "level":
        return level_cost(snapshot) or 0
    if t == "hero_power":
        return _int((_hero_power(snapshot) or {}).get("cost"), 0)
    return 0


def _onehot(value, choices) -> List[float]:
    return [1.0 if value == c else 0.0 for c in choices]


def _slot(v) -> float:
    return (v + 1) / 10.0 if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0.0


def _card_block(card: Optional[Dict]) -> np.ndarray:
    from ml.board_features import TRIBES, minion_from_snapshot
    out = np.zeros(_CARD_DIM, dtype=np.float64)
    if not card:
        return out
    emb, byname, _ = _resources()
    m = minion_from_snapshot(card, byname)
    if m is None:
        return out
    if m["name"] in emb:
        out[:EMB_DIM] = emb[m["name"]]
    out[EMB_DIM:EMB_DIM + 7] = [m["atk"] / 10.0, m["health"] / 10.0, m["tier"] / 6.0,
                                float(m["golden"]), float(m["divine"]),
                                float(m["reborn"]), float(m["taunt"])]
    for tr in m["tribes"]:
        if tr in TRIBES:
            out[EMB_DIM + 7 + TRIBES.index(tr)] = 1.0
    out[-1] = 1.0 if is_spell(card) else 0.0
    return out


def encode_option(snapshot, option: Dict) -> np.ndarray:
    """Fixed OPTION_DIM float32 vector for one option dict."""
    src, tgt = option.get("source") or {}, option.get("target") or {}
    pos = option.get("position")
    gold = _int(_get(snapshot, "gold"), 0)
    cost = option_cost(snapshot, option)
    vidx = min(variant_index(snapshot, option), 4)
    parts = [
        _onehot(option.get("type"), OPTION_TYPES),
        _card_block(option_card(snapshot, option)),
        _onehot(src.get("zone") or "none", ZONES), [_slot(src.get("slot"))],
        _onehot(tgt.get("zone") or "none", ZONES), [_slot(tgt.get("slot"))],
        [1.0 if _slot(pos) else 0.0, _slot(pos)],
        [cost / 10.0, (gold - cost) / 10.0],
        [1.0 if option.get("choice_kind") else 0.0],
        [1.0 if vidx else 0.0, vidx / 4.0],
    ] + _v3_option(snapshot, option, cost, vidx) + _v4_option(snapshot, option, cost) \
        + _v5_option(option)
    return np.concatenate([np.asarray(p, dtype=np.float64) for p in parts]).astype(np.float32)


def _rank(values: List[float], v: float) -> float:
    """Share of `values` strictly below v (0 for an empty list)."""
    return sum(1 for x in values if x < v) / len(values) if values else 0.0


def _v3_option(snapshot, option: Dict, cost: int, vidx: int) -> List[List[float]]:
    """bc-enc-v3 option features (see the module docstring)."""
    ctx = _context(snapshot)
    t = option.get("type")
    src = option.get("source") or {}
    slot, pos = src.get("slot"), option.get("position")
    gold, tier, board = ctx["gold"], ctx["tier"], ctx["board"]
    card = option_card(snapshot, option) or {}
    affordable = [1.0 if gold >= cost else 0.0]
    buy = [0.0] * 5
    if t == "buy" and isinstance(slot, int) and slot >= 0:
        shop = ctx["shop"]
        buy[0] = buy_cost(snapshot, slot) / 3.0
        if slot < len(shop):
            buy[1] = 1.0 if ctx["pair"][slot] else 0.0
            buy[2] = 1.0 if ctx["triple"][slot] else 0.0
            buy[3] = 1.0 if ctx["match"][slot] else 0.0
            buy[4] = (ctx["tiers"][slot] - tier) / 6.0
    shopctx = [0.0] * 5
    if t in ("freeze", "reroll"):
        tiers = ctx["tiers"]
        shopctx = [1.0 if _get(snapshot, "shop_frozen") else 0.0,
                   sum(ctx["triple"]) / MAX_BOARD, sum(ctx["pair"]) / MAX_BOARD,
                   sum(ctx["match"]) / MAX_BOARD,
                   (max(tiers) - tier) / 6.0 if tiers else 0.0]
    posb = [0.0] * 6
    if isinstance(pos, int) and not isinstance(pos, bool) and pos >= 0 and t in ("reposition", "play"):
        mi = _minion_info(card) or {}
        atk, hp = mi.get("atk", 0), mi.get("health", 0)
        if t == "reposition" and isinstance(slot, int):
            last = len(board) - 1
            others_a = [a for i, a in enumerate(ctx["atks"]) if i != slot]
            others_h = [h for i, h in enumerate(ctx["hps"]) if i != slot]
            delta = (pos - slot) / 6.0
        else:
            last = len(board)
            others_a, others_h, delta = ctx["atks"], ctx["hps"], 0.0
        posb = [delta, 1.0 if pos == 0 else 0.0, 1.0 if pos == last else 0.0,
                len(board) / MAX_BOARD, _rank(others_a, atk), _rank(others_h, hp)]
    kw = [1.0 if str(_tags(card).get(k, "")) not in ("", "0") else 0.0 for k in KEYWORDS]
    variant = id_hash(option.get("card_id")) if vidx else [0.0] * HASH_BUCKETS
    return [affordable, buy, shopctx, posb, kw, variant]


def _v4_option(snapshot, option: Dict, cost: int) -> List[List[float]]:
    """bc-enc-v4 option features: hero power identity (the option's own card id,
    else the snapshot's power) on hero_power options; spends the last gold;
    can still buy afterwards."""
    ctx = _context(snapshot)
    gold = ctx["gold"]
    hp_id = [0.0] * HP_VOCAB_DIM
    if option.get("type") == "hero_power":
        hp_id = hero_power_onehot(option.get("card_id")
                                  or (_hero_power(snapshot) or {}).get("card_id"))
    costs = ctx["costs"]
    min_buy = min(costs) if costs else BUY_COST
    return [hp_id, [1.0 if cost > 0 and gold == cost else 0.0,
                    1.0 if gold - cost >= min_buy else 0.0]]


def _v5_option(option: Dict) -> List[List[float]]:
    """bc-enc-v5 option features: which hero a hero-pick offer is.

    Only choice_kind "hero" is filled (skins go through canonical_hero_id).
    Every other option, including a discover that is not a hero pick, is zeros:
    the hero already in play is on the state vector, not repeated per option.
    """
    if option.get("choice_kind") == "hero":
        return [hero_onehot(option.get("card_id"))]
    return [[0.0] * HERO_VOCAB_DIM]


def describe_option(snapshot, option: Dict) -> str:
    """One NEXT line in the advisor's wording (actions.Action.describe style)."""
    t = option.get("type")
    card = option_card(snapshot, option)
    name = (card or {}).get("name") or option.get("card_id") or "?"
    pos = option.get("position")
    if t == "buy":
        if card is not None and is_spell(card):
            return f"Buy spell: {name} ({option_cost(snapshot, option)}g)"
        return f"Buy {name}"
    if t == "sell":
        return f"Sell {name}"
    if t == "reroll":
        return "Roll the shop"
    if t == "freeze":
        return "Freeze the shop"
    if t == "level":
        tier = _int(_get(snapshot, "tavern_tier"), 1) or 1
        return f"Tier up to {tier + 1} ({option_cost(snapshot, option)}g)"
    if t == "hero_power":
        cost = option_cost(snapshot, option)
        return f"Use hero power: {name}" + (f" ({cost}g)" if cost else "")
    if t == "play" and is_dark_gift(option):
        cost = option_cost(snapshot, option)
        return "Dark Gift" + (f" ({cost}g)" if cost else "")
    if t == "play" and is_activate(option):
        cost = option_cost(snapshot, option)
        return f"Activate {name}" + (f" ({cost}g)" if cost else "")
    if t == "play":
        where = (f" at slot {pos + 1}"
                 if isinstance(pos, int) and _zone(snapshot, "board") else "")
        variant = option.get("card_id")
        choice = (f" (Choose One: {variant})"
                  if variant_index(snapshot, option) and variant else "")
        return f"Play {name} from hand{where}{choice}"
    if t == "reposition":
        return f"Move {name} to slot {pos + 1}" if isinstance(pos, int) else f"Move {name}"
    if t == "discover":
        return f"PICK {name}"
    return "End turn"