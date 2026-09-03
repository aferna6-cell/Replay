"""validator tests — pure functions, no network, no LLM. Candidates are built
from the real `actions.Action` / `game_value.WholeGameRec` shapes so the
tests exercise the actual normalization path `director.py` will feed it."""

from hsbg_coach.actions import Action, BUY, END, LEVEL, ROLL, SELL
from hsbg_coach.game_value import WholeGameRec
from hsbg_coach.validator import classify_move, validate


def _buy(name, cost=3):
    return WholeGameRec(Action(BUY, name, cost, {"minion": {"name": name}}),
                        placement=3.0, reason="a fine buy", gain=0.3)


def _sell(name):
    return WholeGameRec(Action(SELL, name, -1, {"minion": {"name": name}}),
                        placement=3.5, reason="frees a slot", gain=-0.1)


def _roll():
    return WholeGameRec(Action(ROLL, cost=1), placement=3.4, reason="roll for value",
                        gain=0.0)


def _level():
    return WholeGameRec(Action(LEVEL, cost=5, detail={"to_tier": 3}),
                        placement=3.2, reason="tier up", gain=0.1)


def _end():
    return WholeGameRec(Action(END), placement=3.9, reason="pass the turn", gain=-0.2)


SNAPSHOT_WITH_GOLD = {"turn": 5, "gold": 4, "tavern_tier": 3}
SNAPSHOT_NO_GOLD = {"turn": 5, "gold": 0, "tavern_tier": 3}


# --- classify_move ---------------------------------------------------------

def test_classify_move_buy_and_sell():
    assert classify_move("Buy Monstrous Macaw") == (BUY, "Monstrous Macaw")
    assert classify_move("Sell Deflect-o-Bot") == (SELL, "Deflect-o-Bot")


def test_classify_move_generic_kinds():
    assert classify_move("End turn")[0] == END
    assert classify_move("Roll the shop")[0] == ROLL
    assert classify_move("Tier up to 4 (7g)")[0] == LEVEL
    assert classify_move("Freeze the shop")[0] == "freeze"


# --- accepts a good buy ------------------------------------------------

def test_accepts_a_good_buy_matching_a_candidate():
    candidates = [_buy("Monstrous Macaw"), _roll(), _end()]
    suggestion = {"move": "Buy Monstrous Macaw", "why": "on-tribe upgrade"}
    accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, candidates)
    assert reason is None
    assert accepted["move"] == "Buy Monstrous Macaw"


def test_corrects_card_name_casing_to_match_candidate():
    candidates = [_buy("Monstrous Macaw")]
    suggestion = {"move": "Buy monstrous macaw", "why": "on-tribe upgrade"}
    accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, candidates)
    assert reason is None
    assert accepted["move"] == "Buy Monstrous Macaw"


def test_accepts_a_good_sell_matching_a_candidate():
    candidates = [_sell("Deflect-o-Bot"), _end()]
    suggestion = {"move": "Sell Deflect-o-Bot", "why": "dead weight"}
    accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, candidates)
    assert reason is None


# --- NO-FLOAT-GOLD -----------------------------------------------------

def test_rejects_end_turn_with_unspent_gold_and_no_justification():
    candidates = [_buy("Monstrous Macaw"), _roll(), _end()]
    suggestion = {"move": "End turn", "why": "board is fine"}
    accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, candidates)
    assert accepted is None
    assert "no-float-gold" in reason


def test_allows_end_turn_with_gold_when_justified():
    candidates = [_buy("Monstrous Macaw"), _roll(), _end()]
    for justification in ("infinite_economy", "deliberate_freeze", "save_for_spike"):
        suggestion = {"move": "End turn", "why": "saving for next turn",
                      "justification": justification}
        accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, candidates)
        assert reason is None, f"expected accept for {justification}"
        assert accepted["move"] == "End turn"


def test_rejects_end_turn_with_bogus_justification_value():
    candidates = [_buy("Monstrous Macaw"), _roll(), _end()]
    suggestion = {"move": "End turn", "why": "whatever",
                  "justification": "because I felt like it"}
    accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, candidates)
    assert accepted is None
    assert "no-float-gold" in reason


def test_allows_end_turn_with_gold_when_nothing_is_affordable():
    # Gold=1 but no candidate costs <=1 (buy costs 3, no roll offered) -> not
    # a float-gold bug, nothing useful to do with the last gold.
    snapshot = {"turn": 5, "gold": 1, "tavern_tier": 3}
    candidates = [_buy("Monstrous Macaw", cost=3), _end()]
    suggestion = {"move": "End turn", "why": "nothing affordable"}
    accepted, reason = validate(suggestion, snapshot, candidates)
    assert reason is None


def test_allows_end_turn_with_zero_gold():
    candidates = [_buy("Monstrous Macaw"), _end()]
    suggestion = {"move": "End turn", "why": "out of gold"}
    accepted, reason = validate(suggestion, SNAPSHOT_NO_GOLD, candidates)
    assert reason is None


# --- hallucinated moves --------------------------------------------------

def test_rejects_hallucinated_buy_not_in_candidates():
    candidates = [_buy("Monstrous Macaw"), _roll(), _end()]
    suggestion = {"move": "Buy Ghost Card That Was Never Offered", "why": "great buy"}
    accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, candidates)
    assert accepted is None
    assert "hallucinated buy" in reason


def test_rejects_hallucinated_sell_not_on_board():
    candidates = [_buy("Monstrous Macaw"), _end()]
    suggestion = {"move": "Sell Card Not On Board", "why": "prune it"}
    accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, candidates)
    assert accepted is None
    assert "hallucinated sell" in reason


def test_rejects_buy_of_card_missing_from_kb_names():
    candidates = [_buy("Totally Fake Minion")]
    kb_names = {"Monstrous Macaw", "Deflect-o-Bot"}
    suggestion = {"move": "Buy Totally Fake Minion", "why": "seems good"}
    accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, candidates, kb_names)
    assert accepted is None
    assert "not in the card KB" in reason


def test_accepts_buy_when_kb_names_contains_it():
    candidates = [_buy("Monstrous Macaw")]
    kb_names = {"Monstrous Macaw", "Deflect-o-Bot"}
    suggestion = {"move": "Buy Monstrous Macaw", "why": "solid"}
    accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, candidates, kb_names)
    assert reason is None


def test_accepts_generic_actions_without_needing_a_candidate_match():
    for move in ("Roll the shop", "Freeze the shop", "Tier up to 4"):
        suggestion = {"move": move, "why": "generic action"}
        accepted, reason = validate(suggestion, SNAPSHOT_NO_GOLD, [])
        assert reason is None, f"{move} should be a legal generic action"


def test_rejects_unrecognized_free_text_move():
    suggestion = {"move": "Do a barrel roll and dab", "why": "why not"}
    accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, [_end()])
    assert accepted is None
    assert "unrecognized move" in reason


def test_accepts_reposition_when_it_matches_a_candidate_description():
    candidates = [WholeGameRec(Action("reposition"), placement=3.0,
                               reason="better attack order", gain=0.1)]
    suggestion = {"move": "Reposition the board", "why": "better order"}
    accepted, reason = validate(suggestion, SNAPSHOT_NO_GOLD, candidates)
    assert reason is None


# --- experiment / hypothesis ---------------------------------------------

def test_rejects_experiment_without_hypothesis():
    candidates = [_buy("Monstrous Macaw")]
    suggestion = {"move": "Buy Monstrous Macaw", "why": "off-meta line",
                  "experiment": True}
    accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, candidates)
    assert accepted is None
    assert "hypothesis" in reason


def test_accepts_experiment_with_hypothesis():
    candidates = [_buy("Monstrous Macaw")]
    suggestion = {"move": "Buy Monstrous Macaw", "why": "off-meta line",
                  "experiment": True, "hypothesis": "this synergy is underrated"}
    accepted, reason = validate(suggestion, SNAPSHOT_WITH_GOLD, candidates)
    assert reason is None
    assert accepted["experiment"] is True


# --- malformed input -------------------------------------------------------

def test_rejects_suggestion_without_move_field():
    accepted, reason = validate({"why": "no move given"}, SNAPSHOT_WITH_GOLD, [])
    assert accepted is None
    assert "no non-empty" in reason


def test_rejects_non_dict_suggestion():
    accepted, reason = validate(["not", "a", "dict"], SNAPSHOT_WITH_GOLD, [])
    assert accepted is None
    assert "not a JSON object" in reason
