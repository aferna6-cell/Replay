"""Battlegrounds TURN: player-level counter wins over GameEntity half-turns."""
from hsbg_coach.parser import parse_line
from hsbg_coach.state import GameState

_GE = "D 17:08:32.5 GameState.DebugPrintPower() -     TAG_CHANGE Entity=GameEntity tag=TURN value={}"
_PL = "D 17:08:32.5 GameState.DebugPrintPower() -     TAG_CHANGE Entity=QuirkyTurtle#1118798 tag=TURN value={}"


def _feed(state, lines):
    for ln in lines:
        ev = parse_line(ln)
        if ev:
            state.apply(ev)


def test_half_turn_game_entity_counter_does_not_flip_the_turn():
    # Verbatim shape from a real log: GameEntity 3 (half-turns), player 2.
    s = GameState()
    _feed(s, ["D 17:07:44.2 GameState.DebugPrintPower() - CREATE_GAME",
              _GE.format(1), _PL.format(1)])
    assert s.current_turn == 1
    turns = []
    for ge, pl in ((2, None), (3, 2), (4, None), (5, 3)):
        _feed(s, [_GE.format(ge)] + ([_PL.format(pl)] if pl else []))
        turns.append(s.current_turn)
    assert turns == [1, 2, 2, 3]


def test_game_entity_only_logs_keep_their_turn():
    s = GameState()
    _feed(s, ["D 09:25:43.1 GameState.DebugPrintPower() - CREATE_GAME", _GE.format(3)])
    assert s.current_turn == 3
