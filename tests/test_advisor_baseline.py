"""Eval-net advisor baseline: mapping the advisor's ranked NEXT onto a label
row's server options, the live display filter, and the trainer wiring
(advisor / model_coarse / --compare fallback to the advisor)."""

import gzip
import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from hsbg_coach import encode as enc  # noqa: E402
from hsbg_coach.actions import (ACTIVATE, BUY, BUY_SPELL, DARK_GIFT, END, FREEZE,  # noqa: E402
                                HERO_POWER, LEVEL, REPOSITION, ROLL, SELL, Action)
from ml import advisor_baseline as AB  # noqa: E402

BUILD = 253216


def _m(name, i, cid=None, eid=None, ctype="MINION"):
    return {"entity_id": eid if eid is not None else 100 + i, "card_id": cid or f"C{i}",
            "name": name, "attack": 2, "health": 2, "position": i + 1,
            "tags": {"CARDTYPE": ctype}}


def _snap(**kw):
    d = {"game_counter": 1, "turn": 5, "phase": "recruit", "tavern_tier": 2, "gold": 10,
         "hero_health": 30, "board": [], "shop": [], "shop_spells": [], "hand_spells": [],
         "hand": [], "hero_power": None, "level_cost": None, "activatable": [],
         "dark_gift": None}
    d.update(kw)
    return d


BOARD = [_m("Rotter", 0, "BG36_099", 10), _m("Dup", 1, "DUP", 11), _m("Dup", 2, "DUP", 12)]
SHOP = [_m("Alpha", 0, "SA", 20), _m("Beta", 1, "SB", 21), _m("Beta", 2, "SB", 22)]
SPELLS = [{"name": "Lasso", "card_id": "SP1", "cost": 2, "entity_id": 30}]
HAND = [_m("Leeroy", 0, "HL", 40), _m("Druid", 1, "HD", 41)]
SNAP = _snap(board=BOARD, shop=SHOP, shop_spells=SPELLS, hand=HAND,
             hero_power={"name": "HP", "card_id": "HP1", "cost": 1, "usable": True})
O = enc.make_option
OPTIONS = [
    O("end_turn"), O("reroll"), O("freeze"), O("level"),
    O("buy", "SA", ("shop", 0)), O("buy", "SB", ("shop", 1)), O("buy", "SB", ("shop", 2)),
    O("buy", "SP1", ("shop", 3)),
    O("sell", "BG36_099", ("board", 0)), O("sell", "DUP", ("board", 1)),
    O("sell", "DUP", ("board", 2)),
    O("hero_power", "HP1", target=("board", 0)), O("hero_power", "HP1", target=("shop", 1)),
    O("play", "BG36_099", ("board", 0), target=("board", 1)),          # activate
    O("play", "BG36_Button_DarkGift"),                                   # dark gift
    O("play", "HL", ("hand", 0), position=0), O("play", "HL", ("hand", 0), position=3),
    O("play", "HDt", ("hand", 1), position=1), O("play", "HDt2", ("hand", 1), position=1),
    O("reposition", "BG36_099", ("board", 0), position=2),
]


def _idx(kind, ref=None, **extra):
    return AB.match_options(SNAP, OPTIONS, dict({"kind": kind, "ref": ref}, **extra))


def test_mapper_every_move_type():
    assert _idx(END) == [0] and _idx(ROLL) == [1] and _idx(FREEZE) == [2] and _idx(LEVEL) == [3]
    assert _idx(BUY, SHOP[0]) == [4]
    assert _idx(BUY, SHOP[2]) == [6]                                     # by entity id
    assert _idx(BUY_SPELL, SPELLS[0]) == [7]                             # slot len(shop)+j
    assert _idx(SELL, BOARD[2]) == [10]
    assert _idx(HERO_POWER, SNAP["hero_power"]) == [11, 12]              # any target
    assert _idx(ACTIVATE, {"card_id": "BG36_099", "entity_id": 10}) == [13]
    assert _idx(DARK_GIFT, {"card_id": "BG36_Button_DarkGift"}) == [14]
    assert _idx("hand_play", HAND[0], slot=0) == [15, 16]                # any position
    assert _idx(REPOSITION) == [19]


def test_mapper_ambiguity_and_unmappable():
    beta_no_eid = {"card_id": "SB", "name": "Beta"}                      # card-id fallback
    hit = _idx(BUY, beta_no_eid)
    assert hit == [5, 6] and AB.is_ambiguous(OPTIONS, hit)
    variants = _idx("hand_play", HAND[1], slot=1)                        # Choose One variants
    assert variants == [17, 18] and AB.is_ambiguous(OPTIONS, variants)
    assert not AB.is_ambiguous(OPTIONS, _idx("hand_play", HAND[0], slot=0))
    assert not AB.is_ambiguous(OPTIONS, _idx(HERO_POWER, SNAP["hero_power"]))
    assert _idx(BUY, {"card_id": "NOPE", "entity_id": 999}) == []
    assert AB.match_options(SNAP, [O("end_turn")], {"kind": ROLL, "ref": None}) == []


def test_action_move_reads_action_detail():
    mv = AB.action_move(Action(BUY, "Alpha", 3, {"minion": SHOP[0]}))
    assert mv == {"kind": BUY, "ref": SHOP[0]}
    assert AB.action_move(Action(ROLL, cost=1)) == {"kind": ROLL, "ref": None}


class _FakeSession:
    def __init__(self, plan):
        self.plan, self.seen = plan, []

    def ranked_moves(self, row, snap):
        self.seen.append((row["game_id"], row["dp_index"]))
        return self.plan[(row["game_id"], row["dp_index"])], {"hero_script_lead": False}


def test_advisor_results_top1_top3_unmappable_and_game_order():
    def row(g, d, chosen):
        return {"game_id": g, "dp_index": d, "options": OPTIONS, "chosen": chosen}
    rows = [row("g2", 0, 1), row("g1", 1, 7), row("g1", 0, 6), row("g1", 2, 0)]
    plan = {("g2", 0): [{"kind": ROLL, "ref": None}],                               # hit
            ("g1", 1): [{"kind": BUY, "ref": SHOP[0]}, {"kind": LEVEL, "ref": None},
                        {"kind": BUY_SPELL, "ref": SPELLS[0]}],                     # top-3
            ("g1", 0): [{"kind": BUY, "ref": {"card_id": "SB"}}],                   # ambiguous hit
            ("g1", 2): [{"kind": ACTIVATE, "ref": {"card_id": "X", "entity_id": 5}},
                        {"kind": END, "ref": None}]}                                # unmappable #1
    sess = _FakeSession(plan)
    hits, diag = AB.advisor_results(rows, [SNAP] * 4, sess)
    assert hits == [(1.0, 1.0), (0.0, 1.0), (1.0, 1.0), (0.0, 1.0)]
    assert sess.seen == [("g1", 0), ("g1", 1), ("g1", 2), ("g2", 0)]   # per game, dp order
    assert diag["top1_unmappable"] == {ACTIVATE: 1}
    assert diag["top1_unmappable_by_chosen_type"] == {"end_turn": 1}
    assert diag["top1_ambiguous"] == {BUY: 1}
    assert diag["top1_unmappable_rate"] == 0.25 and diag["top1_ambiguous_rate"] == 0.25


def test_session_applies_live_display_filter():
    from hsbg_coach.board_value import HeuristicScorer
    board = [_m("Alleycat", 0, "CFM_315", 10), _m("Scallywag", 1, "BGS_061", 11)]
    snap = _snap(gold=0, board=board, shop=[_m("Alpha", 0, "SA", 20)])
    sess = AB.AdvisorSession(HeuristicScorer())
    moves, info = sess.ranked_moves({"game_id": "g", "dp_index": 0}, snap)
    kinds = [m["kind"] for m in moves]
    assert REPOSITION not in kinds
    assert kinds[0] == END or FREEZE in kinds                         # live advice_lines rule
    sess_hand = AB.AdvisorSession(HeuristicScorer(), hand_lines=True)
    snap_hand = _snap(gold=3, board=board[:1], hand=[_m("Leeroy", 0, "HL", 40)])
    moves, _ = sess_hand.ranked_moves({"game_id": "g", "dp_index": 1}, snap_hand)
    assert moves[0] == {"kind": "hand_play", "slot": 0, "ref": snap_hand["hand"][0],
                        "line": "play"}
    assert AB.AdvisorSession(HeuristicScorer()).ranked_moves(
        {"game_id": "g", "dp_index": 2}, _snap(phase="combat"))[0] == []


# --- trainer wiring ---------------------------------------------------------------
def _rows(n_games=5, per_game=6):
    rows = []
    for g in range(n_games):
        for d in range(per_game):
            snap = _snap(turn=3 + d, gold=4, board=[_m("b", 0, "B0", 10), _m("c", 1, "B1", 11)],
                         shop=[_m("x", 0, "SX", 20), _m("y", 1, "SY", 21)])
            opts = [O("end_turn"), O("reroll"), O("buy", "SX", ("shop", 0)),
                    O("buy", "SY", ("shop", 1)), O("sell", "B0", ("board", 0)),
                    O("sell", "B1", ("board", 1))]
            rows.append({"state": snap, "game_id": f"g{g:03d}", "build": BUILD, "turn": 3 + d,
                         "dp_index": d, "mmr": 6000, "options": opts, "chosen": 1,
                         "weight": 1.0})
    return rows


def _data(tmp_path):
    data = tmp_path / "rows.jsonl.gz"
    with gzip.open(data, "wt", encoding="utf-8") as fh:
        for r in _rows():
            fh.write(json.dumps(r) + "\n")
    return data


def test_trainer_reports_advisor_coarse_and_compare_fallback(tmp_path, monkeypatch):
    from hsbg_coach.board_value import HeuristicScorer
    from ml import train_bc_policy as T
    monkeypatch.setattr(AB, "load_scorer", lambda path=None: HeuristicScorer())
    data, pilot = _data(tmp_path), tmp_path / "pilot"
    rc = T.main(["--data", str(data), "--pilot", "--pilot-dir", str(pilot), "--epochs", "5",
                 "--compare", str(tmp_path / "no_policy_yet.pt")])
    assert rc == 0
    m = json.loads((pilot / "pilot_policy.metrics.json").read_text(encoding="utf-8"))
    h = m["heldout"]
    assert {"model", "model_coarse", "random", "advisor", "compare"} <= set(h)
    for name in ("model", "random", "advisor"):
        assert h[name]["overall"]["n"] == 6 and "reroll" in h[name]["per_type"]
    assert h["compare"] == h["advisor"]
    assert m["compare"]["kind"] == "advisor" and "no BC checkpoint" in m["compare"]["note"]
    assert set(m["compare"]["model_minus_compare"]) == {"top1", "top3"}
    adv = m["advisor_baseline"]
    assert adv["enabled"] and adv["mapping"]["decisions"] == 6 and "hits" not in adv
    assert h["model_coarse"]["overall"]["top1"] >= h["model"]["overall"]["top1"]


def test_trainer_advisor_off_or_missing(tmp_path, monkeypatch):
    from ml import train_bc_policy as T
    data = _data(tmp_path)
    monkeypatch.setattr(AB, "load_scorer", lambda path=None: None)       # no eval net
    rc = T.main(["--data", str(data), "--pilot", "--pilot-dir", str(tmp_path / "a"),
                 "--epochs", "2"])
    m = json.loads((tmp_path / "a" / "pilot_policy.metrics.json").read_text(encoding="utf-8"))
    assert rc == 0 and "advisor" not in m["heldout"] and not m["advisor_baseline"]["enabled"]
    called = []
    monkeypatch.setattr(AB, "load_scorer", lambda path=None: called.append(1))
    rc = T.main(["--data", str(data), "--pilot", "--pilot-dir", str(tmp_path / "b"),
                 "--epochs", "2", "--no-baseline-evalnet"])
    assert rc == 0 and not called
