"""A locked comp that isn't coming together pivots to one that is."""
from hsbg_coach import lobby_playbook as lp
from hsbg_coach.lobby_playbook import comp_by_name, evaluate, pivot_target

LOBBY = ["Beast", "Demon", "Mech", "Undead", "Elemental"]


def _card(name):
    return {"name": name, "attack": 3, "health": 3}


def _cards_of(comp, n, exclude=()):
    ci = comp_by_name(comp)
    out = [c for c in ci.core if c not in exclude and lp.in_live_pool(c)]
    return out[:n]


def _snap(board, turn=8, tier=4, locked_turn=4, plan="Undead - Attack Scaling"):
    return {"phase": "recruit", "turn": turn, "tavern_tier": tier, "gold": 5,
            "available_tribes": LOBBY, "board": [_card(n) for n in board],
            "shop": [], "hand": [], "playbook_plan": plan,
            "playbook_locked_turn": locked_turn}


def _demon_board(n=3):
    undead = set(comp_by_name("Undead - Attack Scaling").core)
    return _cards_of("Demons - Self Damage", n, exclude=undead)


def test_stalled_lock_pivots_to_the_comp_the_board_is_building():
    board = _demon_board(3)
    assert len(board) == 3
    st = evaluate(_snap(board))
    assert st.plan == "Demons - Self Damage"
    assert st.pivot_from == "Undead - Attack Scaling"
    assert "not coming together" in st.pivot_reason


def test_young_lock_holds_unless_another_comp_is_far_ahead():
    board = _demon_board(2)
    # Locked 1 turn ago, alt only 2 ahead: keep the plan.
    st = evaluate(_snap(board, turn=5, locked_turn=4, tier=3))
    assert st.plan == "Undead - Attack Scaling" and st.pivot_from is None
    # Same board, 3+ turns later: stalled → pivot.
    st = evaluate(_snap(board, turn=7, locked_turn=4, tier=4))
    assert st.plan == "Demons - Self Damage"


def test_lock_that_is_coming_together_never_pivots():
    undead = _cards_of("Undead - Attack Scaling", 2)
    board = undead + _demon_board(3)
    st = evaluate(_snap(board, turn=9, locked_turn=4, tier=5))
    assert st.plan == "Undead - Attack Scaling"


def test_pivot_only_to_lobby_comps():
    board = _demon_board(3)
    snap = _snap(board)
    snap["available_tribes"] = ["Beast", "Mech", "Undead", "Elemental", "Pirate"]
    assert pivot_target(snap, comp_by_name("Undead - Attack Scaling"), 4,
                        lobby=snap["available_tribes"]) is None


def test_live_memory_switches_plan_and_records_the_pivot():
    from hsbg_coach.live import LiveCoach
    coach = LiveCoach(power_log="/nonexistent/Power.log")
    coach._plan_game = getattr(coach.tracker.state, "game_counter", None)
    coach._plan, coach._plan_turn = "Undead - Attack Scaling", 4
    snap = _snap(_demon_board(3))
    for k in ("playbook_plan", "playbook_locked_turn"):
        snap.pop(k)
    out = coach._with_playbook(snap)
    assert coach._plan == "Demons - Self Damage"
    assert coach._plan_turn == 8                       # the pivot clock restarts
    assert coach.pivots == [(8, "Undead - Attack Scaling", "Demons - Self Damage")]
    assert out["playbook"]["pivot_from"] == "Undead - Attack Scaling"
    # Next frame, same board: no ping-pong back.
    coach._with_playbook(dict(snap, turn=9))
    assert coach._plan == "Demons - Self Damage" and len(coach.pivots) == 1
