"""Eval-net advisor baseline: map the live advisor's ranked NEXT onto a label
row's server options so the BC policy can be gated against what is live today.

The advisor runs the way the overlay runs it (hsbg_coach/live.py):
game_value.rank_actions(snapshot, kb, hero_ctx, scorer, include_reposition=False)
with the scorer from ml/eval_scorer.load_default_scorer (set_net.pt if present,
else eval_net.pt), a hero context built like LiveCoach._ensure_hero_ctx and the
per-game lobby-playbook PLAN lock of LiveCoach._with_playbook. The ranked list
is then filtered and ordered like live.advice_lines:

  * optionally (hand_lines=True; off by default) free hand minions lead, as
    live.advice_lines prints them above the ranked actions ("Play X from hand",
    "Magnetize X onto Y"; with a full board "Sell <weakest>, then play X", whose
    first move is that sell), in hand order, as structured moves. Off by default
    because those lines are not rank_actions output: they fire for every minion
    held in hand (e.g. a Leeroy kept for later) and would dominate the ranking;
  * Reposition is dropped; Freeze is dropped when the advisor buried it
    ("worth keeping" / "no need to freeze" reasons); End turn is dropped unless
    gold <= 0 and no freeze is kept, in which case it leads.

Hero-script text lines (Lens Case, Gallywix cycle, HSReplay hero guides) have
no structured action and are not in the ranking; `hero_script_lead` counts how
often one would have been the first line on screen.

Mapping (target and position ignored):
  buy       shop slot of the offered minion (entity id, else card id)
  buy_spell shop slot len(shop) + j of the spell (entity id, else card id)
  sell      board slot of the minion (entity id, else card id)
  activate  `play` from the board slot of the minion (entity id, else card id)
  dark_gift `play` with source none
  hero_power every hero_power option (any target)
  roll / freeze / level / end   reroll / freeze / level / end_turn
  reposition every reposition option
  hand play `play` from that hand slot (any position / target / Choose One variant)

A move whose options span more than one (type, card_id, source) key is
"ambiguous" (card-id fallback hit several slots, or Choose One variants); it
counts as a hit when the chosen option is among them. An unmappable #1 move is a
top-1 miss. Top-3 looks at the first three moves of the displayed order; an
unmappable move there just contributes nothing.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EVALNET = os.path.join(REPO, "ml", "eval_net.pt")

_FREEZE_BURY = ("worth keeping", "no need to freeze")


def load_scorer(path: Optional[str] = None):
    """The live eval-net scorer (load_default_scorer) for `path` (default
    ml/eval_net.pt; a set_net.pt next to it wins, as live). None if absent."""
    from ml.eval_scorer import load_default_scorer
    path = path or DEFAULT_EVALNET
    set_path = os.path.join(os.path.dirname(os.path.abspath(path)), "set_net.pt")
    if not os.path.isfile(path) and not os.path.isfile(set_path):
        return None
    return load_default_scorer(model_path=path, set_model_path=set_path)


# --- moves ------------------------------------------------------------------
def _d(m) -> Dict:
    if isinstance(m, dict):
        return m
    return {k: getattr(m, k, None) for k in ("entity_id", "card_id", "name", "tags")}


def _slots(cards: Sequence, ref) -> List[int]:
    """Slots of `ref` in `cards`: by entity id if it has one that matches, else
    every slot with its card id."""
    ref = _d(ref) if ref is not None else {}
    cards = [_d(c) for c in cards]
    eid, cid = ref.get("entity_id"), ref.get("card_id")
    if eid is not None:
        hit = [i for i, c in enumerate(cards) if c.get("entity_id") == eid]
        if hit:
            return hit
    return [i for i, c in enumerate(cards) if cid and c.get("card_id") == cid]


def action_move(action) -> Dict:
    """Advisor Action -> move dict {kind, ref}."""
    detail = getattr(action, "detail", None) or {}
    ref = detail.get("minion") or detail.get("spell") or detail.get("hero_power")
    return {"kind": action.kind, "ref": ref}


def match_options(snapshot, options: List[Dict], move: Dict) -> List[int]:
    """Indexes of the row options this move stands for (target / position ignored)."""
    from hsbg_coach.actions import (ACTIVATE, BUY, BUY_SPELL, DARK_GIFT, END, FREEZE,
                                    HERO_POWER, LEVEL, REPOSITION, ROLL, SELL)
    get = snapshot.get if isinstance(snapshot, dict) else (lambda k: getattr(snapshot, k, None))
    board, shop = list(get("board") or []), list(get("shop") or [])
    kind, ref = move["kind"], move.get("ref")

    def where(type_, zone=None, slots=None, card=None):
        out = []
        for i, o in enumerate(options):
            src = o.get("source") or {}
            if o.get("type") != type_:
                continue
            if zone is not None and (src.get("zone") or "none") != zone:
                continue
            if slots is not None and src.get("slot") not in slots:
                continue
            if card is not None and o.get("card_id") != card:
                continue
            out.append(i)
        return out

    if kind == BUY:
        return where("buy", "shop", _slots(shop, ref))
    if kind == BUY_SPELL:
        return where("buy", "shop", [len(shop) + j for j in _slots(get("shop_spells") or [], ref)])
    if kind == SELL:
        return where("sell", "board", _slots(board, ref))
    if kind == ACTIVATE:
        return where("play", "board", _slots(board, ref))
    if kind == DARK_GIFT:
        return where("play", "none")
    if kind == HERO_POWER:
        cid = _d(ref).get("card_id") if ref is not None else None
        hit = where("hero_power", card=cid) if cid else []
        return hit or where("hero_power")
    if kind == ROLL:
        return where("reroll")
    if kind == FREEZE:
        return where("freeze")
    if kind == LEVEL:
        return where("level")
    if kind == END:
        return where("end_turn")
    if kind == REPOSITION:
        return where("reposition")
    if kind == "hand_play":
        return where("play", "hand", [move["slot"]])
    return []


def is_ambiguous(options: List[Dict], idx: List[int]) -> bool:
    keys = {(options[i].get("type"), options[i].get("card_id"),
             (options[i].get("source") or {}).get("zone") or "none",
             (options[i].get("source") or {}).get("slot")) for i in idx}
    return len(keys) > 1


# --- the advisor -------------------------------------------------------------
class AdvisorSession:
    """Runs the live advisor over one game's rows in dp order, keeping the
    per-game state LiveCoach keeps (hero context key, PLAN lock)."""

    def __init__(self, scorer, kb=None, db=None, hand_lines: bool = False):
        from hsbg_coach import cards
        self.scorer = scorer
        self.kb = kb if kb is not None else cards.load_kb()
        self.db = db
        self.hand_lines = hand_lines
        self._game = object()
        self.hero_ctx, self._ctx_key = None, None
        self._plan, self._plan_trigger = None, None

    def _new_game(self, game) -> None:
        if game != self._game:
            self._game = game
            self.hero_ctx, self._ctx_key = None, None
            self._plan, self._plan_trigger = None, None

    def _ensure_hero_ctx(self, snap: Dict) -> None:
        hero = snap.get("hero") or snap.get("hero_name")
        lobby = tuple(snap.get("available_tribes") or ())
        key = (hero, lobby)
        if not hero or key == self._ctx_key:
            return
        try:
            from hsbg_coach.stats import StatsDB, build_hero_context
            if self.db is None:
                self.db = StatsDB.load()
            self.hero_ctx = build_hero_context(
                hero, self.db, available_tribes=list(lobby) if lobby else None,
                board=snap.get("board"), shop=snap.get("shop"),
                turn=snap.get("turn"), kb=self.kb)
            self._ctx_key = key
        except Exception:
            self.hero_ctx = None

    def _with_playbook(self, snap: Dict) -> Dict:
        try:
            from hsbg_coach.lobby_playbook import evaluate
            st = evaluate(snap, kb=self.kb, locked_plan=self._plan)
        except Exception:
            return snap
        if st.committed and not self._plan:
            self._plan, self._plan_trigger = st.plan, st.trigger
        if self._plan:
            st.trigger = st.trigger or self._plan_trigger
        return dict(snap, playbook=st.to_dict(), playbook_plan=self._plan,
                    playbook_trigger=self._plan_trigger)

    def _hand_moves(self, snap: Dict) -> List[Dict]:
        """Structured version of live._hand_play_lines (same order and rules)."""
        from hsbg_coach import live
        from hsbg_coach.hero_scripts import is_hand_playable_item
        from hsbg_coach.magnetize import best_magnetize_target, is_magnetic
        hand = snap.get("hand") or []
        board = snap.get("board") or []
        full = len(board) >= live._MAX_BG_BOARD
        weakest = live._sell_for_room_target(board, snap, self.kb) if board else None
        moves = []
        for slot, m in enumerate(hand):
            if not live._is_hand_minion(m) or is_hand_playable_item(m):
                continue
            if is_magnetic(m, self.kb) and best_magnetize_target(board, self.kb) is not None:
                moves.append({"kind": "hand_play", "slot": slot, "ref": m, "line": "magnetize"})
                continue
            if full and weakest is not None:
                moves.append({"kind": "sell", "ref": weakest, "line": "sell_then_play"})
            else:
                moves.append({"kind": "hand_play", "slot": slot, "ref": m, "line": "play"})
        return moves

    def ranked_moves(self, row: Dict, snapshot: Dict) -> Tuple[List[Dict], Dict]:
        """(moves in live display order, info) for one row."""
        from hsbg_coach.actions import END, FREEZE, REPOSITION
        from hsbg_coach.game_value import rank_actions
        self._new_game(row.get("game_id"))
        info = {"hero_script_lead": False}
        if snapshot.get("phase") not in ("recruit", "unknown"):
            return [], info
        self._ensure_hero_ctx(snapshot)
        snap = self._with_playbook(snapshot)
        recs, _ = rank_actions(snap, kb=self.kb, hero_ctx=self.hero_ctx,
                               scorer=self.scorer, include_reposition=False)
        try:
            from hsbg_coach.hero_scripts import hero_script_lines
            info["hero_script_lead"] = bool(hero_script_lines(snap, kb=self.kb))
        except Exception:
            pass
        shown, freeze_kept = [], False
        for r in recs:
            kind = r.action.kind
            if kind == REPOSITION or kind == END:
                continue
            if kind == FREEZE:
                if any(tok in (r.reason or "").lower() for tok in _FREEZE_BURY):
                    continue
                freeze_kept = True
            shown.append(r)
        gold = snap.get("gold")
        if gold is not None and int(gold) <= 0 and not freeze_kept:
            end = next((r for r in recs if r.action.kind == END), None)
            if end is not None:
                shown.insert(0, end)
        moves = [action_move(r.action) for r in shown]
        if self.hand_lines:
            moves = self._hand_moves(snap) + moves
        return moves, info


def advisor_results(rows: List[Dict], snapshots: List[Dict], session: AdvisorSession,
                    k: int = 3) -> Tuple[List[Tuple[float, float]], Dict]:
    """Per-row (top1, top3) hits for the advisor, plus mapping diagnostics.
    Rows are processed per game in dp_index order (live keeps per-game state);
    results come back in the input order."""
    order = sorted(range(len(rows)), key=lambda i: (str(rows[i].get("game_id")),
                                                   rows[i].get("dp_index") or 0, i))
    res: List[Optional[Tuple[float, float]]] = [None] * len(rows)
    diag = {"decisions": len(rows), "errors": 0, "hero_script_lead": 0,
            "top1_unmappable": {}, "top1_ambiguous": {}, "top1_moves": {},
            "top1_unmappable_by_chosen_type": {}, "no_moves": 0}

    def bump(d, key):
        d[key] = d.get(key, 0) + 1

    for i in order:
        row, snap = rows[i], snapshots[i]
        opts, chosen = row["options"], row["chosen"]
        try:
            moves, info = session.ranked_moves(row, snap)
        except Exception:
            diag["errors"] += 1
            moves, info = [], {}
        diag["hero_script_lead"] += bool(info.get("hero_script_lead"))
        if not moves:
            diag["no_moves"] += 1
            bump(diag["top1_unmappable_by_chosen_type"], opts[chosen].get("type", "?"))
            res[i] = (0.0, 0.0)
            continue
        matched = [match_options(snap, opts, mv) for mv in moves[:k]]
        kind1 = moves[0]["kind"]
        bump(diag["top1_moves"], kind1)
        if not matched[0]:
            bump(diag["top1_unmappable"], kind1)
            bump(diag["top1_unmappable_by_chosen_type"], opts[chosen].get("type", "?"))
        elif is_ambiguous(opts, matched[0]):
            bump(diag["top1_ambiguous"], kind1)
        t1 = float(chosen in matched[0])
        t3 = float(any(chosen in m for m in matched))
        res[i] = (t1, t3)
    n = max(len(rows), 1)
    moves1 = diag["top1_moves"]
    diag["top1_unmappable_rate_by_move"] = {
        kd: round(diag["top1_unmappable"].get(kd, 0) / c, 4) for kd, c in sorted(moves1.items())}
    diag["top1_ambiguous_rate_by_move"] = {
        kd: round(diag["top1_ambiguous"].get(kd, 0) / c, 4) for kd, c in sorted(moves1.items())}
    diag["top1_unmappable_rate"] = round(
        (sum(diag["top1_unmappable"].values()) + diag["no_moves"]) / n, 4)
    diag["top1_ambiguous_rate"] = round(sum(diag["top1_ambiguous"].values()) / n, 4)
    return res, diag
