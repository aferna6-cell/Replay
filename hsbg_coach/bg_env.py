"""Phase 0 recruit-phase simulator — the self-play RL environment.

This is the make-or-break component from ``specs/self-play-rl-agent.md`` §2: a
deliberately *simplified* 8-player Battlegrounds lobby that is faithful on the
layers that shape decisions — the economy (gold curve, buy/sell/roll/freeze,
tavern-up discounts), shop generation from a finite shared pool, triples with a
discover reward, and real combat resolved by ``sim.py`` — while deferring
heroes, trinkets, anomalies and the battlecry long tail.

The card pool is curated: minions from the committed knowledge base
(`data/cards/bg_cards.json`) whose combat behaviour the simulator models
(stats + Divine Shield / Taunt / Poisonous / Reborn / Windfury / Cleave tags,
plus the deathrattle/start-of-combat registry in ``effects.py``). Cards with a
card2vec embedding are preferred so every pool minion is meaningful to the
learned models.

Two consumers, one env:
  * **Dataset generation** — play scripted lobbies and label every per-turn
    state with the final placement it led to (the mid-game training data the
    eval net was missing; see ``ml/midgame_dataset.py``).
  * **RL** — a gym-style ``reset()/step()`` API with a fixed discrete action
    space and a legal-action mask, driven from seat 0 while scripted (or
    league) policies drive the other seats (``ml/train_ppo.py``).

Stdlib only, like the rest of the core package.
"""

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import cards as cards_mod
from .pace import STANDARD_TAVERN_TIER, load_pace, _at as _curve_at
from .sim import Combatant, simulate_once

# --- rules constants ---------------------------------------------------------
MAX_BOARD = 7
MAX_HAND = 10
MAX_TIER = 6
BUY_COST = 3
SELL_VALUE = 1
ROLL_COST = 1
START_HP = 30
MAX_TURNS = 20
SHOP_SLOTS = {1: 3, 2: 4, 3: 4, 4: 5, 5: 5, 6: 6}
# Phase 2N-C: current Battlegrounds reference uses 7 copies at Tier 6.
POOL_COPIES = {1: 15, 2: 15, 3: 13, 4: 11, 5: 9, 6: 7}
UPGRADE_COST = {1: 5, 2: 7, 3: 8, 4: 9, 5: 10}
VALID_SCALING_MODES = frozenset({"ratio", "residual"})
N_LOBBY_TRIBES = 5

# Phase 2N lifecycle toggles (independent; default ON after 2N-B).
PHASE_2N_DEATH_RETURN = True   # eliminated players return board/hand/shop
PHASE_2N_FREEZE_TOPUP = True   # incomplete frozen shops refill to SHOP_SLOTS

TRIBES = ["Beast", "Mech", "Murloc", "Dragon", "Demon", "Elemental",
          "Pirate", "Naga", "Undead", "Quilboar"]

# --- fixed discrete action space (mask tells which are legal) -----------------
# 0..6   buy shop slot i           (shop has at most 6 slots; 7 keeps it uniform)
# 7..16  play hand slot i
# 17..23 sell board slot i
# 24 roll · 25 tier up · 26 freeze (toggle) · 27 end turn
A_BUY0, N_BUY = 0, 7
A_PLAY0, N_PLAY = 7, MAX_HAND
A_SELL0, N_SELL = 17, MAX_BOARD
A_ROLL, A_LEVEL, A_FREEZE, A_END = 24, 25, 26, 27
N_ACTIONS = 28

_KW_TO_TAG = {
    "DIVINE_SHIELD": "DIVINE_SHIELD", "TAUNT": "TAUNT",
    "POISONOUS": "POISONOUS", "VENOMOUS": "POISONOUS",
    "REBORN": "REBORN", "WINDFURY": "WINDFURY",
    "MEGA_WINDFURY": "WINDFURY", "CLEAVE": "CLEAVE",
}


def gold_at(turn: int) -> int:
    return min(10, turn + 2)


@dataclass
class EnvMinion:
    card_id: str
    name: str
    tier: int
    attack: int
    health: int
    tribes: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    golden: bool = False

    def tags(self) -> Dict[str, str]:
        t = {kw_tag: "1" for kw, kw_tag in _KW_TO_TAG.items() if kw in self.keywords}
        t["TECH_LEVEL"] = str(self.tier)
        if self.golden:
            t["PREMIUM"] = "1"
        return t

    def view(self) -> Dict:
        """The dict shape the rest of the package reads (MinionView-like)."""
        return {"name": self.name, "card_id": self.card_id,
                "attack": self.attack, "health": self.health,
                "tags": self.tags()}

    def as_golden(self) -> "EnvMinion":
        return EnvMinion(self.card_id, self.name, self.tier,
                         self.attack * 2, self.health * 2,
                         list(self.tribes), list(self.keywords), golden=True)


def build_pool(kb: Optional[Dict] = None, emb_names: Optional[set] = None,
               lobby_tribes: Optional[Sequence[str]] = None
               ) -> List[EnvMinion]:
    """The curated minion catalogue (one entry per distinct card).

    Prefers cards that have a card2vec embedding (meta-relevant, meaningful to
    the learned models); tiers that would end up too thin fall back to the full
    knowledge base. Restricted to the lobby's tribes + tribeless/All minions.
    """
    kb = kb if kb is not None else cards_mod.load_kb()
    allowed = set(lobby_tribes or TRIBES)

    def in_lobby(ck) -> bool:
        trs = ck.tribes or []
        return (not trs) or ("All" in trs) or any(t in allowed for t in trs)

    def usable(ck) -> bool:
        return (ck.tier is not None and 1 <= ck.tier <= MAX_TIER
                and ck.attack is not None and ck.health is not None
                and "_SKIN_" not in ck.card_id and in_lobby(ck))

    # Distinct by name (the KB carries duplicate printings of some minions).
    all_ok: Dict[str, object] = {}
    for ck in kb.values():
        if usable(ck) and ck.name not in all_ok:
            all_ok[ck.name] = ck
    pref = {n: ck for n, ck in all_ok.items()
            if emb_names is None or n in emb_names}

    catalogue: List[EnvMinion] = []
    for tier in range(1, MAX_TIER + 1):
        chosen = [ck for ck in pref.values() if ck.tier == tier]
        if len(chosen) < 6:                       # too thin — widen to the full KB
            chosen = [ck for ck in all_ok.values() if ck.tier == tier]
        for ck in chosen:
            catalogue.append(EnvMinion(ck.card_id, ck.name, ck.tier,
                                       int(ck.attack), int(ck.health),
                                       list(ck.tribes), list(ck.keywords)))
    return catalogue


@dataclass
class PlayerState:
    idx: int
    board: List[EnvMinion] = field(default_factory=list)
    hand: List[EnvMinion] = field(default_factory=list)
    shop: List[EnvMinion] = field(default_factory=list)
    gold: int = 3
    tier: int = 1
    hp: int = START_HP
    frozen: bool = False
    turns_since_level: int = 0
    alive: bool = True
    placement: Optional[int] = None
    last_board: List[EnvMinion] = field(default_factory=list)  # ghost fights

    def level_cost(self) -> Optional[int]:
        if self.tier >= MAX_TIER:
            return None
        return max(0, UPGRADE_COST[self.tier] - self.turns_since_level)

    def strength(self) -> int:
        return sum(m.attack + m.health for m in self.board)


class BGEnv:
    """8-player recruit-phase lobby. Seat 0 is the learning agent."""

    def __init__(self, n_players: int = 8, seed: Optional[int] = None,
                 opponent_policies: Optional[Sequence[Callable]] = None,
                 kb: Optional[Dict] = None, emb_names: Optional[set] = None,
                 combat_runs: int = 1,
                 scaling_mode: str = "residual"):
        if scaling_mode not in VALID_SCALING_MODES:
            raise ValueError(
                f"scaling_mode must be one of {sorted(VALID_SCALING_MODES)}, "
                f"got {scaling_mode!r}")
        self.n_players = n_players
        self.rng = random.Random(seed)
        self._kb = kb if kb is not None else cards_mod.load_kb()
        self._emb_names = emb_names
        self.opponent_policies = list(opponent_policies or [])
        self.combat_runs = combat_runs
        self.scaling_mode = scaling_mode
        self.turn = 0
        self.players: List[PlayerState] = []
        self.lobby_tribes: List[str] = []
        self._pool: Dict[str, int] = {}
        self._catalogue: Dict[str, EnvMinion] = {}
        self._done = True
        self._scaling = load_pace().get("scaling", {})
        self._agent_actions = 0
        # Optional observational hook: (env, player, meta_dict) -> None.
        # Must not mutate env. Used by Phase 2M shop/pool audit only.
        self.pool_deal_hook: Optional[Callable] = None
        self._deal_reason: str = "unknown"

    MAX_ACTIONS_PER_TURN = 40                  # same cap scripted seats get

    # -- lifecycle -------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict:
        if seed is not None:
            self.rng = random.Random(seed)
        self.lobby_tribes = self.rng.sample(TRIBES, N_LOBBY_TRIBES)
        catalogue = build_pool(self._kb, self._emb_names, self.lobby_tribes)
        self._catalogue = {m.name: m for m in catalogue}
        self._pool = {m.name: POOL_COPIES[m.tier] for m in catalogue}
        self.players = [PlayerState(idx=i) for i in range(self.n_players)]
        self.turn = 1
        self._done = False
        self._agent_actions = 0
        for p in self.players:
            p.gold = gold_at(self.turn)
            self._deal_reason = "reset"
            self._deal_shop(p)
        return self.observe(0)

    # -- shop / pool -----------------------------------------------------------
    def _names_at_or_below(self, tier: int) -> List[str]:
        return [n for n, m in self._catalogue.items()
                if m.tier <= tier and self._pool.get(n, 0) > 0]

    def _draw(self, tier: int) -> Optional[EnvMinion]:
        names = self._names_at_or_below(tier)
        if not names:
            return None
        weights = [self._pool[n] for n in names]
        name = self.rng.choices(names, weights=weights, k=1)[0]
        self._pool[name] -= 1
        base = self._catalogue[name]
        return EnvMinion(base.card_id, base.name, base.tier, base.attack,
                         base.health, list(base.tribes), list(base.keywords))

    def _return_to_pool(self, m: EnvMinion) -> None:
        # Golden minions were built from 3 copies; return them all.
        self._pool[m.name] = self._pool.get(m.name, 0) + (3 if m.golden else 1)

    def _return_player_holdings_to_pool(self, p: PlayerState) -> None:
        """Return board, hand, and shop copies to the shared pool (Phase 2N-B)."""
        for m in list(p.board) + list(p.hand) + list(p.shop):
            self._return_to_pool(m)
        p.board = []
        p.hand = []
        p.shop = []

    def _deal_shop(self, p: PlayerState) -> None:
        """Deal shop slots. Optional ``pool_deal_hook`` is observational only."""
        if p.frozen:
            p.frozen = False
            if PHASE_2N_FREEZE_TOPUP:
                # Keep frozen minions; top up to current tier slot count.
                target = int(SHOP_SLOTS[p.tier])
                track = getattr(self, "_pool_audit_track_names", None)
                pre_draw_remaining: Dict[str, int] = {}
                eligible = self._names_at_or_below(p.tier)
                eligible_total = sum(self._pool[n] for n in eligible)
                if track is not None:
                    pre_draw_remaining = {
                        n: int(self._pool.get(n, 0)) for n in track
                        if n in self._catalogue}
                elif self.pool_deal_hook is not None:
                    pre_draw_remaining = {
                        n: int(c) for n, c in self._pool.items()}
                dealt_names: List[str] = [m.name for m in p.shop]
                n_new = 0
                while len(p.shop) < target:
                    m = self._draw(p.tier)
                    if m is None:
                        break
                    p.shop.append(m)
                    dealt_names.append(m.name)
                    n_new += 1
                if self.pool_deal_hook is not None:
                    self.pool_deal_hook(self, p, {
                        "reason": self._deal_reason,
                        "frozen_skip": n_new == 0,
                        "freeze_topup": True,
                        "tavern_tier": p.tier,
                        "n_slots": n_new,
                        "dealt_names": dealt_names,
                        "card_remaining": pre_draw_remaining,
                        "eligible_total_copies": int(eligible_total),
                    })
                return
            # Legacy: preserve shop verbatim, no top-up.
            if self.pool_deal_hook is not None:
                self.pool_deal_hook(self, p, {
                    "reason": self._deal_reason,
                    "frozen_skip": True,
                    "tavern_tier": p.tier,
                    "n_slots": 0,
                    "dealt_names": [m.name for m in p.shop],
                    "card_remaining": {},
                    "eligible_total_copies": 0,
                })
            return
        for m in p.shop:
            self._return_to_pool(m)
        p.shop = []
        # Snapshot *after* return-to-pool, *before* draws (exact live draw state).
        pre_draw_remaining = {}
        eligible = self._names_at_or_below(p.tier)
        eligible_total = sum(self._pool[n] for n in eligible)
        n_slots = int(SHOP_SLOTS[p.tier])
        track = getattr(self, "_pool_audit_track_names", None)
        if track is not None:
            pre_draw_remaining = {n: int(self._pool.get(n, 0)) for n in track
                                  if n in self._catalogue}
        elif self.pool_deal_hook is not None:
            # Default: snapshot all catalogue remaining counts (heavy; prefer track).
            pre_draw_remaining = {n: int(c) for n, c in self._pool.items()}

        dealt_names = []
        for _ in range(n_slots):
            m = self._draw(p.tier)
            if m is not None:
                p.shop.append(m)
                dealt_names.append(m.name)

        if self.pool_deal_hook is not None:
            self.pool_deal_hook(self, p, {
                "reason": self._deal_reason,
                "frozen_skip": False,
                "tavern_tier": p.tier,
                "n_slots": n_slots,
                "dealt_names": dealt_names,
                "card_remaining": pre_draw_remaining,
                "eligible_total_copies": int(eligible_total),
            })

    # -- observation / mask ------------------------------------------------------
    def observe(self, seat: int) -> Dict:
        p = self.players[seat]
        alive = [q for q in self.players if q.alive]
        opp_strengths = [q.strength() for q in alive if q.idx != seat]
        return {
            "turn": self.turn,
            "gold": p.gold,
            "tavern_tier": p.tier,
            "hero_health": p.hp,
            "level_cost": p.level_cost(),
            "frozen": p.frozen,
            "players_alive": len(alive),
            "max_opp_strength": max(opp_strengths, default=0),
            "board": [m.view() for m in p.board],
            "hand": [m.view() for m in p.hand],
            "shop": [m.view() for m in p.shop],
            "lobby_tribes": list(self.lobby_tribes),
        }

    def snapshot(self, seat: int = 0) -> Dict:
        """State in the advisor's snapshot shape (so scorers/search plug in)."""
        o = self.observe(seat)
        o["notes"] = []
        return o

    def legal_mask(self, seat: int = 0) -> List[bool]:
        p = self.players[seat]
        mask = [False] * N_ACTIONS
        for i in range(len(p.shop)):
            if p.gold >= BUY_COST and len(p.hand) < MAX_HAND:
                mask[A_BUY0 + i] = True
        for i in range(len(p.hand)):
            if len(p.board) < MAX_BOARD:
                mask[A_PLAY0 + i] = True
        for i in range(len(p.board)):
            mask[A_SELL0 + i] = True
        if p.gold >= ROLL_COST:
            mask[A_ROLL] = True
        lc = p.level_cost()
        if lc is not None and p.gold >= lc:
            mask[A_LEVEL] = True
        if p.shop:
            mask[A_FREEZE] = True
        mask[A_END] = True
        return mask

    # -- recruit actions ----------------------------------------------------------
    def _apply(self, seat: int, action: int) -> bool:
        """Apply one recruit action. Returns True when the seat's turn ends."""
        p = self.players[seat]
        if not self.legal_mask(seat)[action]:
            return False                          # illegal = no-op (masked anyway)
        if A_BUY0 <= action < A_BUY0 + N_BUY:
            m = p.shop.pop(action - A_BUY0)
            p.gold -= BUY_COST
            p.hand.append(m)
            self._check_triple(p, m.name)
        elif A_PLAY0 <= action < A_PLAY0 + N_PLAY:
            m = p.hand.pop(action - A_PLAY0)
            p.board.append(m)
            self._check_triple(p, m.name)
        elif A_SELL0 <= action < A_SELL0 + N_SELL:
            m = p.board.pop(action - A_SELL0)
            p.gold += SELL_VALUE
            self._return_to_pool(m)
        elif action == A_ROLL:
            p.gold -= ROLL_COST
            p.frozen = False
            self._deal_reason = "roll"
            self._deal_shop(p)
        elif action == A_LEVEL:
            p.gold -= p.level_cost()
            p.tier += 1
            p.turns_since_level = 0
        elif action == A_FREEZE:
            p.frozen = not p.frozen
        elif action == A_END:
            return True
        return False

    def _check_triple(self, p: PlayerState, name: str) -> None:
        """3 non-golden copies across board+hand merge into a golden + discover."""
        owned = [(zone, i) for zone in ("board", "hand")
                 for i, m in enumerate(getattr(p, zone)) if m.name == name
                 and not m.golden]
        if len(owned) < 3:
            return
        base = None
        for zone, i in sorted(owned[:3], key=lambda t: -t[1]):
            base = getattr(p, zone).pop(i)
        if len(p.hand) < MAX_HAND:
            p.hand.append(base.as_golden())
        # Discover reward: a minion one tier up (simplified: random draw).
        reward_tier = min(MAX_TIER, p.tier + 1)
        if len(p.hand) < MAX_HAND:
            reward = self._draw(reward_tier)
            if reward is not None:
                p.hand.append(reward)

    # -- end-of-turn scaling ---------------------------------------------------------
    # Real BG boards grow exponentially (buffs, battlecries, end-of-turn effects —
    # the measured `scaling` pace curve, ~1.5-2x per turn late). The Phase 0 card
    # pool models combat keywords but not the buff long tail, so boards would only
    # grow by bought stats and the whole lobby's damage/elimination dynamics would
    # be wrong. Bridge the gap abstractly at end of recruit.
    #
    # Simulator v1 (`ratio`): multiply the whole board by the real turn-to-turn
    # growth ratio — tends to double-count recruit purchases that already moved stats.
    #
    # Simulator v1.1 (`residual`, default): apply only the missing budget between
    # the Firestone target for this turn and stats already produced by recruit.
    def _scaling_growth_factor(self, p: PlayerState) -> float:
        exp_tier = STANDARD_TAVERN_TIER.get(self.turn, 6.0)
        deficit = max(0.0, exp_tier - p.tier)
        factor = max(0.3, 0.98 + 0.035 * p.tier - 0.32 * deficit)
        if p.turns_since_level == 0:
            factor *= 0.6
        factor *= self.rng.uniform(0.88, 1.14)
        return factor

    def _end_of_turn_scaling_ratio(self, p: PlayerState) -> None:
        if not p.board:
            return
        prev = _curve_at(self._scaling, max(1, self.turn - 1)) or 1.0
        cur = _curve_at(self._scaling, self.turn) or prev
        ratio = (cur / prev) if prev else 1.0
        g = ratio * self._scaling_growth_factor(p)
        g = max(1.0, g)
        for m in p.board:
            m.attack = max(1, round(m.attack * g))
            m.health = max(1, round(m.health * g))

    def _end_of_turn_scaling_residual(self, p: PlayerState) -> None:
        """Apply only unexplained abstract growth once boards exceed the pace curve.

        Before turn 10, behave like ratio scaling (recruit growth has not yet
        compounded enough to double-count). From turn 10 onward, subtract any
        board stats already above the Firestone pace target from the ratio-mode
        abstract buff budget instead of multiplying the full board again.
        """
        if not p.board:
            return
        current = p.strength()
        prev = _curve_at(self._scaling, max(1, self.turn - 1)) or 1.0
        cur = _curve_at(self._scaling, self.turn) or prev
        ratio = (cur / prev) if prev else 1.0
        factor = self._scaling_growth_factor(p)
        ratio_g = max(1.0, ratio * factor)
        ratio_add = current * (ratio_g - 1)
        if self.turn >= 10:
            pace_target = cur * factor
            over = max(0.0, current - pace_target)
            residual_add = max(0.0, ratio_add - over)
        else:
            residual_add = ratio_add
        if residual_add <= 0:
            return
        for m in p.board:
            share = (m.attack + m.health) / current
            add = residual_add * share
            total = m.attack + m.health
            if total <= 0:
                continue
            m.attack = max(1, round(m.attack + add * m.attack / total))
            m.health = max(1, round(m.health + add * m.health / total))

    def _end_of_turn_scaling(self, p: PlayerState) -> None:
        if self.scaling_mode == "ratio":
            self._end_of_turn_scaling_ratio(p)
        elif self.scaling_mode == "residual":
            self._end_of_turn_scaling_residual(p)
        else:                                          # pragma: no cover
            raise ValueError(f"unknown scaling_mode: {self.scaling_mode!r}")

    def _scale_all(self) -> None:
        for p in self.players:
            if p.alive:
                self._end_of_turn_scaling(p)

    # -- combat --------------------------------------------------------------------
    def _combatants(self, board: List[EnvMinion]) -> List[Combatant]:
        return [Combatant.from_minion(m.view()) for m in board]

    @staticmethod
    def _hero_damage(raw: int, tier: int, board: List[EnvMinion]) -> int:
        """Real BG damage is winner tier + the *tiers* of surviving minions;
        sim.py reports winner tier + survivor *count*. Recover the count and
        weight it by the winner's average minion tier so late-game losses hit
        like they do on ladder (that's what ends real games by ~turn 14)."""
        survivors = max(1, abs(raw) - tier)
        avg_tier = (sum(m.tier for m in board) / len(board)) if board else 1.0
        return tier + max(1, round(survivors * avg_tier))

    def _run_combat(self) -> None:
        alive = [p for p in self.players if p.alive]
        order = alive[:]
        self.rng.shuffle(order)
        pairs: List[Tuple[PlayerState, Optional[PlayerState]]] = []
        for i in range(0, len(order) - 1, 2):
            pairs.append((order[i], order[i + 1]))
        if len(order) % 2 == 1:
            pairs.append((order[-1], None))       # fights a ghost

        dead_boards = [p.last_board for p in self.players
                       if not p.alive and p.last_board]
        for a, b in pairs:
            if b is None:
                if not dead_boards:
                    continue
                ghost = self.rng.choice(dead_boards)
                ghost_tier = max((g.tier for g in ghost), default=1)
                dmg = simulate_once(self._combatants(a.board),
                                    self._combatants(ghost), self.rng,
                                    tier_a=a.tier, tier_b=ghost_tier)
                if dmg < 0:
                    a.hp -= self._hero_damage(dmg, ghost_tier, ghost)
                continue
            dmg = simulate_once(self._combatants(a.board),
                                self._combatants(b.board), self.rng,
                                tier_a=a.tier, tier_b=b.tier)
            if dmg > 0:
                b.hp -= self._hero_damage(dmg, a.tier, a.board)
            elif dmg < 0:
                a.hp -= self._hero_damage(dmg, b.tier, b.board)

        # Deaths → placements (weakest dead this round takes the worst slot).
        dead = [p for p in alive if p.hp <= 0]
        if dead:
            place = len(alive)
            for p in sorted(dead, key=lambda x: (x.hp, x.strength())):
                p.alive = False
                p.placement = place
                p.last_board = [EnvMinion(m.card_id, m.name, m.tier, m.attack,
                                          m.health, list(m.tribes),
                                          list(m.keywords), m.golden)
                                for m in p.board]
                if PHASE_2N_DEATH_RETURN:
                    self._return_player_holdings_to_pool(p)
                place -= 1

    def _finalize(self) -> None:
        survivors = sorted((p for p in self.players if p.alive),
                           key=lambda p: (-p.hp, -p.strength()))
        for i, p in enumerate(survivors):
            p.placement = i + 1
            p.alive = False
        self._done = True

    def _start_turn(self) -> None:
        self.turn += 1
        for p in self.players:
            if not p.alive:
                continue
            p.gold = gold_at(self.turn)
            p.turns_since_level += 1
            self._deal_reason = "start_turn"
            self._deal_shop(p)

    def _scripted_seat(self, seat: int) -> None:
        p = self.players[seat]
        if not p.alive:
            return
        policy = (self.opponent_policies[seat - 1]
                  if 0 <= seat - 1 < len(self.opponent_policies) else None)
        policy = policy or greedy_policy
        for _ in range(40):                        # hard cap per recruit turn
            a = policy(self.observe(seat), self.legal_mask(seat), self.rng)
            if self._apply(seat, a):
                break

    def _advance(self) -> None:
        """Other seats act, combat resolves, next turn begins."""
        for seat in range(1, self.n_players):
            self._scripted_seat(seat)
        self._scale_all()
        self._run_combat()
        alive = [p for p in self.players if p.alive]
        if len(alive) <= 1 or self.turn >= MAX_TURNS:
            self._finalize()
            return
        if not self.players[0].alive:
            self._done = True
            return
        self._start_turn()

    # -- gym-style step ---------------------------------------------------------------
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Apply seat-0's action. Ending the turn advances the whole lobby."""
        if self._done:
            raise RuntimeError("episode is done — call reset()")
        self._agent_actions += 1
        turn_over = self._apply(0, action)
        if self._agent_actions >= self.MAX_ACTIONS_PER_TURN:
            turn_over = True                   # stalling policy: force end turn
        if turn_over:
            self._agent_actions = 0
            self._advance()
        me = self.players[0]
        if self._done or me.placement is not None:
            self._done = True
            placement = me.placement or 1
            reward = (4.5 - placement) / 3.5       # zero-mean: 1st=+1, 8th=-1
            return self.observe(0), reward, True, {"placement": placement}
        return self.observe(0), 0.0, False, {}

    @property
    def done(self) -> bool:
        return self._done

    # -- scripted full-lobby playout (dataset generation) ------------------------------
    def play_scripted(self, policies: Optional[Sequence[Callable]] = None,
                      recruit_tracer: Optional[object] = None
                      ) -> List[Dict]:
        """Play the whole lobby with scripted seats (seat 0 included).

        Returns one record per (seat, turn): the seat's end-of-recruit state
        (advisor snapshot shape) labeled with the final placement it led to.

        Optional ``recruit_tracer`` receives observational callbacks; it must
        not mutate env state (Phase 2C measurement-only tracing).
        """
        self.reset()
        policies = list(policies or [])
        records: List[Dict] = []
        lobby_id = getattr(recruit_tracer, "lobby_id", 0)
        if recruit_tracer is not None and hasattr(recruit_tracer, "begin_lobby"):
            recruit_tracer.begin_lobby(
                lobby_id, lobby_id, list(self.lobby_tribes))
        while not self._done:
            for seat in range(self.n_players):
                p = self.players[seat]
                if not p.alive:
                    continue
                policy = (policies[seat] if seat < len(policies) else None
                          ) or greedy_policy
                shop_generation = 0
                if recruit_tracer is not None and hasattr(recruit_tracer, "begin_seat_recruit"):
                    recruit_tracer.begin_seat_recruit(seat, self.turn, p)
                for _ in range(40):
                    obs = self.observe(seat)
                    mask = self.legal_mask(seat)
                    if recruit_tracer is not None and hasattr(recruit_tracer, "before_action"):
                        recruit_tracer.before_action(
                            seat, self.turn, shop_generation, obs, mask)
                    a = policy(obs, mask, self.rng)
                    ended = self._apply(seat, a)
                    if recruit_tracer is not None and hasattr(recruit_tracer, "after_action"):
                        recruit_tracer.after_action(
                            seat, self.turn, shop_generation, a, ended, p)
                    if a == A_ROLL:
                        shop_generation += 1
                    if ended:
                        break
                if recruit_tracer is not None and hasattr(recruit_tracer, "end_seat_recruit"):
                    recruit_tracer.end_seat_recruit(seat, self.turn, p)
                records.append({"seat": seat, "turn": self.turn,
                                "state": self.snapshot(seat)})
            self._scale_all()
            self._run_combat()
            alive = [p for p in self.players if p.alive]
            if len(alive) <= 1 or self.turn >= MAX_TURNS:
                self._finalize()
            else:
                self._start_turn()
        for r in records:
            r["placement"] = self.players[r["seat"]].placement
        if recruit_tracer is not None and hasattr(recruit_tracer, "end_lobby"):
            recruit_tracer.end_lobby(self.players)
        return records


# --- scripted baseline policies ----------------------------------------------------
def _tribe_counts(minions: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in minions:
        for t in (m.get("tribes") or []):
            counts[t] = counts.get(t, 0) + 1
    return counts


def random_policy(obs: Dict, mask: List[bool], rng: random.Random) -> int:
    """Uniform over legal actions, mildly biased away from instantly ending."""
    legal = [i for i, ok in enumerate(mask) if ok and i != A_END]
    if legal and rng.random() < 0.8:
        return rng.choice(legal)
    return A_END


def make_greedy_policy(level_bias: float = 0.0) -> Callable:
    """Greedy baseline with a pacing dial: positive bias levels ahead of the
    curve (greedy-leveler), negative stays on tempo. Used to diversify the
    scripted field so generated datasets cover different pacing styles."""
    def policy(obs: Dict, mask: List[bool], rng: random.Random) -> int:
        return _greedy(obs, mask, rng, level_bias)
    return policy


def greedy_policy(obs: Dict, mask: List[bool], rng: random.Random) -> int:
    """Play hand, level on the pace curve, buy the biggest affordable minion,
    roll surplus gold. The baseline every learned policy must beat."""
    return _greedy(obs, mask, rng, 0.0)


def build_aware_greedy_policy(obs: Dict, mask: List[bool],
                              rng: random.Random) -> int:
    """Greedy recruit policy with build-path buy valuation (Phase 2D treatment).

    Identical to ``greedy_policy`` except legally buyable minions are ranked by
    ``build_aware_buy_score`` instead of raw ``attack + health``.
    """
    from .build_aware_policy import build_aware_buy_score
    return _greedy(obs, mask, rng, 0.0, buy_scorer=build_aware_buy_score)


def seeded_core_stress_greedy_policy(obs: Dict, mask: List[bool],
                                     rng: random.Random) -> int:
    """Phase 2E oracle/stress policy — force missing cores when seeded."""
    from .seeded_core_stress_policy import seeded_core_buy_override
    return _greedy(obs, mask, rng, 0.0, buy_override=seeded_core_buy_override)


def seeded_core_deploy_stress_greedy_policy(obs: Dict, mask: List[bool],
                                            rng: random.Random) -> int:
    """Phase 2G oracle — Phase 2E buy oracle + board-slot sell for hand cores."""
    from .seeded_core_deploy_policy import seeded_core_deploy_sell_action
    from .seeded_core_stress_policy import seeded_core_buy_override
    sell = seeded_core_deploy_sell_action(obs, mask)
    if sell is not None:
        return sell
    return _greedy(obs, mask, rng, 0.0, buy_override=seeded_core_buy_override)


def _greedy(obs: Dict, mask: List[bool], rng: random.Random,
            level_bias: float,
            buy_scorer: Optional[Callable[[Dict, int], float]] = None,
            buy_override: Optional[Callable[[Dict, List[bool], List[int]],
                                            Optional[int]]] = None) -> int:
    if buy_scorer is None:
        from .build_aware_policy import raw_stat_buy_score
        buy_scorer = raw_stat_buy_score
    if any(mask[A_PLAY0:A_PLAY0 + N_PLAY]):
        return A_PLAY0 + next(i for i in range(N_PLAY) if mask[A_PLAY0 + i])
    target = STANDARD_TAVERN_TIER.get(obs["turn"], 6.0) + level_bias
    if mask[A_LEVEL] and obs["tavern_tier"] < target - 0.45:
        return A_LEVEL
    buys = [i for i in range(len(obs["shop"])) if mask[A_BUY0 + i]]
    if buys:
        if buy_override is not None:
            pick = buy_override(obs, mask, buys)
            if pick is not None:
                return A_BUY0 + pick
        if len(obs["board"]) + len(obs["hand"]) < MAX_BOARD + 1:
            return A_BUY0 + max(buys, key=lambda i: buy_scorer(obs, i))
    # Board full: upgrade — sell the weakest if the best shop minion beats it.
    if buys and len(obs["board"]) >= MAX_BOARD:
        if buy_override is not None:
            pick = buy_override(obs, mask, buys)
            best = pick if pick is not None else max(buys, key=lambda i: buy_scorer(obs, i))
        else:
            best = max(buys, key=lambda i: buy_scorer(obs, i))
        bval = buy_scorer(obs, best)
        weakest = min(range(len(obs["board"])),
                      key=lambda i: (obs["board"][i].get("attack") or 0)
                      + (obs["board"][i].get("health") or 0))
        wval = ((obs["board"][weakest].get("attack") or 0)
                + (obs["board"][weakest].get("health") or 0))
        if bval > wval and mask[A_SELL0 + weakest]:
            return A_SELL0 + weakest
    if mask[A_ROLL] and obs["gold"] >= BUY_COST + ROLL_COST:
        return A_ROLL
    return A_END


def pace_curves(lobbies: int = 100, seed: int = 0,
                policy: Callable = greedy_policy) -> Dict[int, Dict[str, float]]:
    """Average tavern tier + board stats by turn under a scripted policy —
    the validation hook: compare against data/stats/firestone_pace.json."""
    sums: Dict[int, List[float]] = {}
    for i in range(lobbies):
        env = BGEnv(seed=seed + i)
        for rec in env.play_scripted([policy] * 8):
            s = rec["state"]
            row = sums.setdefault(rec["turn"], [0.0, 0.0, 0])
            row[0] += s["tavern_tier"]
            row[1] += sum((m.get("attack") or 0) + (m.get("health") or 0)
                          for m in s["board"])
            row[2] += 1
        del env
    return {t: {"tier": v[0] / v[2], "stats": v[1] / v[2], "n": v[2]}
            for t, v in sorted(sums.items())}


def main(argv=None):                               # pragma: no cover - CLI glue
    import argparse
    from .pace import load_pace, _at
    ap = argparse.ArgumentParser(description="Validate the Phase 0 env vs real pace")
    ap.add_argument("--lobbies", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    pace = load_pace()
    curves = pace_curves(a.lobbies, a.seed)
    print(f"{'turn':>4} {'env tier':>9} {'real tier':>10} {'env stats':>10} "
          f"{'real stats':>11}")
    for t, row in curves.items():
        rt = _at(pace.get("tavern_tier", {}), t)
        rs = _at(pace.get("scaling", {}), t)
        print(f"{t:>4} {row['tier']:>9.2f} {rt if rt else float('nan'):>10.2f} "
              f"{row['stats']:>10.0f} {rs if rs else float('nan'):>11.0f}")
    return 0


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(main())
