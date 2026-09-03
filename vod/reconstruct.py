"""Frame reads -> recorder-schema trajectory records.

Input: time-ordered frame reads (phase + structured state). Output: one record
per recruit turn, shaped exactly like hsbg_coach/recorder.py lines so
ml/board_dataset.trajectory_examples ingests them unchanged:

    {"game_id", "state": <Snapshot-ish dict>, "action_type": "vod_turn",
     "action_detail": {"inferred_actions": [...]}, "placement": int|None,
     "source": "vod"}

Design choices that keep this trainable even when vision is imperfect:
  * The eval net needs only (state.board, context, placement) — action
    inference is best-effort garnish for the future policy net, never a gate.
  * Per turn we keep the LAST confident read (end-of-turn board ≈ what the
    recorder snapshots at combat start).
  * A turn-number drop or an endscreen splits games; a game with no endscreen
    placement is still emitted (placement None -> training skips it, the
    states remain for later backfill).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

MIN_CONFIDENCE = 0.5


@dataclass
class FrameRead:
    ts: float                       # seconds into the VOD
    phase: str                      # recruit | combat | endscreen | other
    state: Optional[Dict] = None    # STATE_SCHEMA payload for read frames


def _minion_dict(m: Dict, pos: int) -> Dict:
    tags = {"PREMIUM": "1"} if m.get("golden") else {}
    return {"entity_id": 0, "card_id": None, "name": m.get("name"),
            "attack": m.get("attack"), "health": m.get("health"),
            "position": pos, "tags": tags}


def _snapshot(read: Dict, turn: Optional[int]) -> Dict:
    """STATE_SCHEMA payload -> Snapshot-shaped dict (recorder parity)."""
    return {
        "game_counter": 0, "turn": turn, "phase": "recruit",
        "tavern_tier": read.get("tavern_tier"), "gold": read.get("gold"),
        "hero_health": read.get("hero_health"),
        "board": [_minion_dict(m, i)
                  for i, m in enumerate(read.get("board") or [])],
        "shop": [_minion_dict(m, i)
                 for i, m in enumerate(read.get("shop") or [])],
        "shop_spells": [], "hand_spells": [], "hero_power": None,
        "anomaly": None, "level_cost": None, "trinkets": [],
        "opponent_profiles": [], "hero": None,
        "hero_name": read.get("hero_name"), "hand": [],
        "opponents_seen": [], "notes": ["reconstructed from VOD"],
    }


def _names(state: Dict) -> List[str]:
    return [m["name"] for m in state.get("board", []) if m.get("name")]


def infer_actions(prev: Optional[Dict], cur: Dict) -> List[Dict]:
    """Best-effort board diff between consecutive turns."""
    if prev is None:
        return []
    actions: List[Dict] = []
    pt, ct = prev.get("tavern_tier"), cur.get("tavern_tier")
    if pt and ct and ct > pt:
        actions.append({"type": "tier_up", "to": ct})
    before, after = _names(prev), _names(cur)
    counts: Dict[str, int] = {}
    for n in before:
        counts[n] = counts.get(n, 0) - 1
    for n in after:
        counts[n] = counts.get(n, 0) + 1
    for name, delta in counts.items():
        kind = "play" if delta > 0 else "sell"
        for _ in range(abs(delta)):
            actions.append({"type": kind, "name": name})
    return actions


@dataclass
class _Game:
    turns: List[Dict] = field(default_factory=list)   # snapshot dicts
    placement: Optional[int] = None
    hero_name: Optional[str] = None


def _segment_games(reads: List[FrameRead]) -> List[_Game]:
    """Split the read stream into games on turn resets / endscreens, keeping
    the last confident recruit read per turn."""
    games: List[_Game] = [_Game()]
    cur_turn: Optional[int] = None
    pending: Optional[Dict] = None   # last confident read of the current turn

    def flush_turn():
        nonlocal pending
        if pending is not None:
            games[-1].turns.append(_snapshot(pending, cur_turn))
            if pending.get("hero_name") and not games[-1].hero_name:
                games[-1].hero_name = pending["hero_name"]
        pending = None

    def next_game():
        flush_turn()
        if games[-1].turns or games[-1].placement is not None:
            games.append(_Game())

    for r in reads:
        if r.phase == "endscreen":
            flush_turn()
            if r.state and r.state.get("final_placement"):
                games[-1].placement = int(r.state["final_placement"])
            next_game()
            cur_turn = None
            continue
        if r.phase != "recruit" or not r.state:
            continue
        if (r.state.get("confidence") or 0) < MIN_CONFIDENCE:
            continue
        turn = r.state.get("turn")
        if turn is not None:
            if cur_turn is not None and turn < cur_turn:   # reset => new game
                next_game()
            if cur_turn is not None and turn != cur_turn:
                flush_turn()
            cur_turn = turn
        pending = r.state
    flush_turn()
    return [g for g in games if g.turns]


def reconstruct(reads: List[FrameRead], vod_id: str) -> List[List[Dict]]:
    """Full pipeline: reads -> list of games -> recorder-schema records."""
    out: List[List[Dict]] = []
    for gi, game in enumerate(_segment_games(reads), start=1):
        records: List[Dict] = []
        prev: Optional[Dict] = None
        for snap in game.turns:
            records.append({
                "game_id": f"vod-{vod_id}-g{gi}",
                "state": snap,
                "action_type": "vod_turn",
                "action_detail": {"inferred_actions": infer_actions(prev, snap)},
                "placement": game.placement,
                "source": "vod",
            })
            prev = snap
        out.append(records)
    return out
