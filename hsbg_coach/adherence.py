"""Adherence check: replay real Power.logs and measure how closely the game
followed the lobby playbook — and whether that tracked with placement.

Two separate questions, reported separately:

  * **Coach adherence** — did the coach's own NEXT follow its rules? At every
    recruit-phase state change the replay asks the live coach for NEXT and
    flags: a Naga mention; a buy whose own reason says "never buy"; after the
    PLAN lock, an off-plan buy or a Roll as
    NEXT while an affordable comp card sits in the shop, or selling a named
    PLAN card; Hero Power as NEXT on a hero whose HSReplay guide calls the
    hero power weak.
  * **Player adherence** — did the game actually follow the plan? After the
    lock, per turn: when a comp card was in the shop, was one bought (hunt);
    how many buys were off-plan (discipline); how many named comp cards made
    the final board (final comp). Hero pick vs. the coach's hero pick too.

Placement comes from the log. The summary splits games at the median player
adherence and compares average placement / top-4 rate — the signal for "is
the plan worth following", which needs many games before it means anything.

Nothing here trains a model. Adherence is the wrong training target: a model
rewarded for matching the plan learns to copy the rules, not to win.
Placement stays the objective; adherence is a diagnostic next to it.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import asdict, dataclass, field
from statistics import mean, median
from typing import Dict, Iterable, List, Optional

_BUY = re.compile(r"^Buy (?!spell:)(.+?)(?: \(| —|$)")
_SELL = re.compile(r"^Sell (.+?)(?: \(| —|$)")
_ROLL = re.compile(r"^Roll the shop")
_HP = re.compile(r"^(?:Use hero power|Hero Power —)")
_FINAL_PIECES = 4            # named comp cards on the final board = full credit


@dataclass
class TurnAudit:
    turn: Optional[int]
    tier: Optional[int] = None
    plan: Optional[str] = None
    next_line: Optional[str] = None           # coach NEXT at the turn's first frame
    frames: int = 0
    violations: List[str] = field(default_factory=list)
    plan_cards_up: List[str] = field(default_factory=list)   # comp cards seen in shop
    bought: List[str] = field(default_factory=list)
    bought_on_plan: List[str] = field(default_factory=list)
    bought_off_plan: List[str] = field(default_factory=list)


@dataclass
class GameAudit:
    log: str
    game: int
    hero: Optional[str] = None
    hero_offer: List[str] = field(default_factory=list)
    coach_hero: Optional[str] = None
    lobby: List[str] = field(default_factory=list)
    lock_turn: Optional[int] = None
    lock_comp: Optional[str] = None
    lock_trigger: Optional[str] = None
    pivots: List[list] = field(default_factory=list)       # [turn, from, to]
    final_plan: Optional[str] = None
    final_board: List[str] = field(default_factory=list)
    final_plan_pieces: int = 0
    placement: Optional[int] = None      # final place — only when the log reached game over
    finished: bool = False
    turns: List[TurnAudit] = field(default_factory=list)

    # -- derived -----------------------------------------------------------
    @property
    def hero_followed(self) -> Optional[bool]:
        if not self.hero or not self.coach_hero:
            return None
        return self.hero == self.coach_hero

    def _post_lock(self) -> List[TurnAudit]:
        return [t for t in self.turns if t.plan]

    @property
    def hunt_rate(self) -> Optional[float]:
        up = [t for t in self._post_lock() if t.plan_cards_up]
        if not up:
            return None
        return sum(1 for t in up if t.bought_on_plan) / len(up)

    @property
    def discipline(self) -> Optional[float]:
        buys = sum(len(t.bought) for t in self._post_lock())
        if not buys:
            return None
        off = sum(len(t.bought_off_plan) for t in self._post_lock())
        return 1.0 - off / buys

    @property
    def final_comp(self) -> Optional[float]:
        if not self.final_plan:
            return None
        return min(1.0, self.final_plan_pieces / _FINAL_PIECES)

    @property
    def player_score(self) -> Optional[float]:
        """0-100: mean of hunt rate, buy discipline and final-comp completion."""
        parts = [x for x in (self.hunt_rate, self.discipline, self.final_comp)
                 if x is not None]
        return round(100 * mean(parts), 1) if parts else None

    @property
    def coach_frames(self) -> int:
        return sum(t.frames for t in self.turns)

    @property
    def coach_violations(self) -> List[str]:
        return [v for t in self.turns for v in t.violations]

    @property
    def coach_score(self) -> Optional[float]:
        """0-100: share of NEXT frames with no rule violation."""
        n = self.coach_frames
        return round(100 * (1 - len(self.coach_violations) / n), 1) if n else None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("hero_followed", "hunt_rate", "discipline", "final_comp",
                  "player_score", "coach_score"):
            d[k] = getattr(self, k)
        d["coach_violations"] = self.coach_violations
        return d


# ---------------------------------------------------------------------------
# Log discovery
# ---------------------------------------------------------------------------

def find_logs(paths: Iterable[str]) -> List[str]:
    """Power.log files from files / directories / globs (dirs are searched for
    ``Power.log`` and per-launch ``Hearthstone_*/Power.log``), oldest first."""
    out: List[str] = []
    for p in paths:
        for q in (glob.glob(p) or [p]):
            if os.path.isfile(q):
                out.append(q)
            elif os.path.isdir(q):
                for pat in ("Power.log", os.path.join("Hearthstone_*", "Power.log"),
                            os.path.join("**", "Power.log")):
                    out += glob.glob(os.path.join(q, pat), recursive=True)
    seen, uniq = set(), []
    for f in out:
        k = os.path.abspath(f)
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return sorted(uniq, key=os.path.getmtime)


def default_logs() -> List[str]:
    """Every Power.log in Hearthstone's log folders on this machine."""
    from . import config
    return find_logs(config.log_dir_candidates())


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def _name(m) -> Optional[str]:
    return m.get("name") if isinstance(m, dict) else getattr(m, "name", None)


def _eid(m):
    return m.get("entity_id") if isinstance(m, dict) else getattr(m, "entity_id", None)


def _state_key(snap: dict) -> tuple:
    return (snap.get("turn"), snap.get("gold"),
            tuple(_name(m) for m in snap.get("shop") or []),
            tuple(_name(m) for m in snap.get("board") or []),
            tuple(_name(m) for m in snap.get("hand") or []))


def _check_frame(snap: dict, lines: List[str], kb) -> List[str]:
    """Rule violations in the coach's NEXT for this frame."""
    from .lobby_playbook import comp_by_name, is_on_plan, is_plan_piece, plan_support
    out: List[str] = []
    top = (lines or [""])[0]
    if any("naga" in (ln or "").lower() for ln in lines[:3]):
        out.append("naga mentioned in NEXT")
    if _BUY.match(top) and re.search(r"never buy|out of the live", top, re.I):
        out.append("NEXT buys a card its own reason says never to buy")
    if _HP.match(top):
        try:
            from .hero_power_verdict import weak_quote
            from .hsreplay_guides import lookup_hero
            if weak_quote(lookup_hero(snap)):
                out.append("weak hero power as NEXT")
        except Exception:
            pass
    ci = comp_by_name(snap.get("playbook_plan"))
    if ci is None:
        return out
    gold = int(snap.get("gold") or 0)
    core_up = [_name(m) for m in snap.get("shop") or [] if _name(m) in ci.core]
    affordable = bool(core_up) and gold >= 3
    m = _BUY.match(top)
    if m:
        target = m.group(1).strip()
        shop = {_name(x): x for x in snap.get("shop") or []}
        support = plan_support(snap, ci)
        if (affordable and target not in core_up and target not in support
                and not is_on_plan(shop.get(target) or target, ci, kb)):
            out.append(f"off-plan buy {target} as NEXT with {core_up[0]} up")
    elif _ROLL.match(top) and affordable:
        out.append(f"roll as NEXT with {core_up[0]} affordable")
    s = _SELL.match(top)
    if s and is_plan_piece(s.group(1).strip(), ci):
        out.append(f"sell PLAN card {s.group(1).strip()} as NEXT")
    return out


def audit_log(path: str, kb=None, max_games: Optional[int] = None) -> List[GameAudit]:
    """Replay one Power.log through the live coach; one GameAudit per game."""
    from .choices import ChoiceParser
    from .lobby_playbook import comp_by_name, is_on_plan, plan_support
    from .live import LiveCoach
    from .parser import parse_line

    coach = LiveCoach(power_log=path, strategy_header=False)
    kb = kb if kb is not None else coach.kb
    coach.choices = ChoiceParser()
    games: List[GameAudit] = []
    g: Optional[GameAudit] = None
    turn: Optional[TurnAudit] = None
    shop_seen: Dict[object, str] = {}         # entity id -> name seen in shop this turn
    last_key = None
    last_snap: Optional[dict] = None

    def close_turn():
        nonlocal turn, shop_seen
        if turn is None or last_snap is None:
            turn, shop_seen = None, {}
            return
        owned = {_eid(m): _name(m) for z in ("board", "hand")
                 for m in last_snap.get(z) or []}
        bought = [shop_seen[e] for e in shop_seen if e in owned]
        turn.bought = [b for b in bought if b]
        ci = comp_by_name(turn.plan)
        if ci is not None:
            support = plan_support(last_snap, ci)
            for b in turn.bought:
                (turn.bought_on_plan if (b in ci.cards or b in ci.core or b in support
                                         or is_on_plan(b, ci, kb))
                 else turn.bought_off_plan).append(b)
        g.turns.append(turn)
        turn, shop_seen = None, {}

    def close_game():
        nonlocal g
        if g is None:
            return
        close_turn()
        g.pivots = [list(p) for p in coach.pivots]
        g.final_plan = coach._plan
        ci = comp_by_name(g.final_plan)
        if last_snap is not None:
            g.final_board = [_name(m) for m in last_snap.get("board") or []]
        if ci is not None:
            g.final_plan_pieces = sum(1 for n in set(g.final_board) if n in ci.cards)
        if not g.finished:
            g.placement = None             # mid-game leaderboard place, not a result
        if g.turns or g.hero:
            games.append(g)
        g = None

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            offer = coach.choices.feed(line)
            if offer is not None:
                coach._offer = offer
            elif "SendChoices" in line:
                coach._offer = None
            ev = parse_line(line)
            if ev is None:
                continue
            coach._active = True
            coach.tracker.feed(ev)
            coach._version += 1
            gc = coach.tracker.state.game_counter
            if g is not None and gc != g.game:
                close_game()
                if max_games and len(games) >= max_games:
                    break
            if g is None and gc:
                g = GameAudit(log=path, game=gc)
            if g is None:
                continue
            o = coach._offer
            if o is not None and o.kind == "hero" and list(o.names) != g.hero_offer:
                g.hero_offer = list(o.names)
                try:
                    _, _, out = coach.frame()
                    pick = next((ln for ln in out if re.match(r"^\s*1\. ", ln)), None)
                    if pick:
                        g.coach_hero = re.sub(r"^\s*1\. ", "", pick).split(" — ")[0]
                except Exception:
                    pass
                continue
            if coach.tracker.phase.value == "game_over":
                g.finished = True
                p = coach.tracker.placement()
                if p is not None:
                    g.placement = p
            if coach.tracker.phase.value != "recruit":
                if turn is not None:
                    close_turn()
                continue
            snap = coach.tracker.snapshot().to_dict()
            key = _state_key(snap)
            if key == last_key:
                continue
            last_key = key
            try:
                fsnap, _, lines = coach.frame()
            except Exception:
                continue
            last_snap = fsnap
            hero = fsnap.get("hero_name")
            if hero and not re.search(r"BaconPH|_PH\b|^TB_", hero):
                g.hero = hero                # skip hero-select placeholders
            lobby = list(fsnap.get("available_tribes") or [])
            if len(lobby) > len(g.lobby):
                g.lobby = lobby              # tribes reveal over the first turns
            if coach._plan and g.lock_comp is None:
                g.lock_comp, g.lock_turn = coach._plan, fsnap.get("turn")
                g.lock_trigger = coach._plan_trigger
            if turn is not None and turn.turn != fsnap.get("turn"):
                close_turn()
            if turn is None:
                turn = TurnAudit(turn=fsnap.get("turn"), tier=fsnap.get("tavern_tier"),
                                 plan=coach._plan, next_line=(lines or [None])[0])
            turn.plan = coach._plan or turn.plan
            turn.frames += 1
            turn.violations += _check_frame(fsnap, lines, kb)
            ci = comp_by_name(coach._plan)
            for m in fsnap.get("shop") or []:
                shop_seen[_eid(m)] = _name(m)
                if ci is not None and _name(m) in ci.core and _name(m) not in turn.plan_cards_up:
                    turn.plan_cards_up.append(_name(m))
    close_game()
    return games


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def summarize(games: List[GameAudit]) -> dict:
    scored = [g for g in games if g.player_score is not None and g.placement]
    out = {
        "games": len(games),
        "finished": sum(1 for g in games if g.finished),
        "with_placement": sum(1 for g in games if g.placement),
        "committed": sum(1 for g in games if g.lock_comp),
        "pivots": sum(len(g.pivots) for g in games),
        "avg_placement": (round(mean(g.placement for g in games if g.placement), 2)
                          if any(g.placement for g in games) else None),
        "avg_player_score": (round(mean(g.player_score for g in games
                                        if g.player_score is not None), 1)
                             if any(g.player_score is not None for g in games) else None),
        "avg_coach_score": (round(mean(g.coach_score for g in games
                                       if g.coach_score is not None), 1)
                            if any(g.coach_score is not None for g in games) else None),
        "hero_followed": sum(1 for g in games if g.hero_followed),
        "hero_known": sum(1 for g in games if g.hero_followed is not None),
    }
    viol: Dict[str, int] = {}
    for g in games:
        for v in g.coach_violations:
            k = re.sub(r" [A-Z][\w' ,-]+? (?=as NEXT|up|affordable)", " X ", v)
            viol[k] = viol.get(k, 0) + 1
    out["violation_types"] = dict(sorted(viol.items(), key=lambda kv: -kv[1]))
    if len(scored) >= 2:
        cut = median(g.player_score for g in scored)
        hi = [g for g in scored if g.player_score >= cut]
        lo = [g for g in scored if g.player_score < cut] or [g for g in scored if g not in hi]
        out["split"] = {
            "median_player_score": cut,
            "followed": {"games": len(hi), "avg_placement": round(mean(g.placement for g in hi), 2),
                         "top4": round(sum(g.placement <= 4 for g in hi) / len(hi), 2)},
            "strayed": {"games": len(lo), "avg_placement": round(mean(g.placement for g in lo), 2),
                        "top4": round(sum(g.placement <= 4 for g in lo) / len(lo), 2)},
        }
    return out


def render_text(games: List[GameAudit], summary: dict) -> str:
    lines = ["HSBG playbook adherence", "=" * 23, ""]
    hdr = f"{'#':>3}  {'place':>5}  {'player':>6}  {'coach':>5}  {'lock':>9}  {'hunt':>4}  {'disc':>4}  {'final':>5}  hero / plan"
    lines += [hdr, "-" * len(hdr)]
    for i, g in enumerate(games, 1):
        lock = f"t{g.lock_turn}" if g.lock_turn is not None else "never"
        piv = f" (+{len(g.pivots)} pivot)" if g.pivots else ""
        hero = g.hero or "?"
        if g.hero_followed is False:
            hero += f" [coach: {g.coach_hero}]"
        place = g.placement or ("?" if g.finished else "unfin")
        lines.append(
            f"{i:>3}  {place:>5}  "
            f"{'—' if g.player_score is None else g.player_score:>6}  "
            f"{'—' if g.coach_score is None else g.coach_score:>5}  {lock:>9}  "
            f"{_pct(g.hunt_rate):>4}  {_pct(g.discipline):>4}  {_pct(g.final_comp):>5}  "
            f"{hero} / {g.final_plan or '—'}{piv}")
    s = summary
    lines += ["",
              f"games {s['games']} · committed {s['committed']} · pivots {s['pivots']} · "
              f"avg place {s['avg_placement']} · player adherence {s['avg_player_score']} · "
              f"coach adherence {s['avg_coach_score']} · coach hero taken "
              f"{s['hero_followed']}/{s['hero_known']}"]
    if s.get("violation_types"):
        lines += ["", "Coach rule violations (NEXT broke its own playbook):"]
        lines += [f"  {n:>4}  {k}" for k, n in s["violation_types"].items()]
    sp = s.get("split")
    if sp:
        lines += ["", f"Followed the plan (player ≥ {sp['median_player_score']}): "
                      f"{sp['followed']['games']} games, avg place {sp['followed']['avg_placement']}, "
                      f"top-4 {sp['followed']['top4']:.0%}",
                  f"Strayed from the plan:                 {sp['strayed']['games']} games, "
                  f"avg place {sp['strayed']['avg_placement']}, top-4 {sp['strayed']['top4']:.0%}"]
        if s["games"] < 20:
            lines.append(f"  (only {s['games']} games — too few to say whether following "
                         "the plan helps; aim for 20+)")
    lines += ["", "player = mean of hunt (bought a comp card when one was up), disc (share of "
              "on-plan buys after the lock) and final (named comp cards on the final board, "
              f"{_FINAL_PIECES} = full). coach = share of NEXT frames with no rule violation."]
    return "\n".join(lines)
