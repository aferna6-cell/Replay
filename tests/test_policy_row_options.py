"""NEXT policy on server option lists: the trainer's candidate set is each row's
own `options`, the held-out split / --pilot mode, encode_option on every
server option kind (bc-enc-v2), the legal_options parity fixes, and the parity
script."""

import gzip
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from hsbg_coach import encode as enc  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BUILD = 253216


def _m(name, i, atk=2, hp=2, ctype="MINION", cid=None, eid=None, **tags):
    t = {"CARDTYPE": ctype}
    t.update({k: str(v) for k, v in tags.items()})
    return {"entity_id": eid if eid is not None else 100 + i, "card_id": cid or f"TEST_{i}",
            "name": name, "attack": atk, "health": hp, "position": i + 1, "tags": t}


def _snap(**kw):
    d = {"game_counter": 1, "turn": 5, "phase": "recruit", "tavern_tier": 2, "gold": 10,
         "hero_health": 30, "board": [], "shop": [], "shop_spells": [], "hand_spells": [],
         "hand": [], "hero_power": None, "level_cost": None, "activatable": [],
         "dark_gift": None}
    d.update(kw)
    return d


def _of(opts, t, zone=None):
    return [o for o in opts if o["type"] == t
            and (zone is None or (o["source"]["zone"] or "none") == zone)]


# --- encoder version / option kinds ---------------------------------------------
def test_encoder_version_bumped_with_new_option_layout():
    assert enc.ENCODER_VERSION == "bc-enc-v5"
    assert enc.STATE_DIM == (
        3 * enc._ZONE_DIM + len(enc.STATE_SCALARS) + enc.HP_VOCAB_DIM + enc.HERO_VOCAB_DIM)
    assert enc.OPTION_DIM == (
        len(enc.OPTION_TYPES) + enc._CARD_DIM + 2 * (len(enc.ZONES) + 1) + 2 + 2 + 1 + 2
        + enc._V3_OPTION + enc._V4_OPTION + enc._V5_OPTION)


def test_encode_option_separates_every_server_option_kind():
    board = [_m("Brain Rotter", 0, cid="BG36_099", INTERACTABLE_OBJECT=1), _m("b", 1)]
    hand = [_m("Parent", 0, cid="BG31_880", CHOOSE_ONE=1), _m("Spell", 1, ctype="SPELL", cid="SP")]
    snap = _snap(board=board, hand=hand, gold=5,
                 dark_gift={"name": "Dark Discovery", "card_id": "BG36_Button_DarkGift",
                            "cost": 3, "usable": True})
    opts = [
        enc.make_option("play", "BG36_099", ("board", 0)),                        # activate
        enc.make_option("play", "BG36_099", ("board", 0), target=("board", 1)),   # targeted
        enc.make_option("play", "BG36_Button_DarkGift"),                          # dark gift
        enc.make_option("play", "BG31_880t", ("hand", 0), position=1),            # variant 1
        enc.make_option("play", "BG31_880t2", ("hand", 0), position=1),           # variant 2
        enc.make_option("play", "BG31_880", ("hand", 0), position=1),             # parent
        enc.make_option("play", "SP", ("hand", 1)),                               # untargeted
        enc.make_option("play", "SP", ("hand", 1), target=("shop", 0)),
        enc.make_option("discover", "BG36_099", choice_kind="discover"),
    ]
    vecs = [enc.encode_option(snap, o) for o in opts]
    assert all(v.shape == (enc.OPTION_DIM,) and np.isfinite(v).all() for v in vecs)
    assert len({v.tobytes() for v in vecs}) == len(opts)
    hero = dict(opts[6], target_entity=93)          # debug ids stay ignored (live has none)
    assert np.array_equal(enc.encode_option(snap, hero), vecs[6])
    assert enc.variant_index(snap, opts[3]) == 1 and enc.variant_index(snap, opts[4]) == 2
    assert enc.variant_index(snap, opts[5]) == 0
    assert enc.option_cost(snap, opts[2]) == 3                     # Dark Gift cost
    assert enc.describe_option(snap, opts[2]) == "Dark Gift (3g)"
    assert enc.describe_option(snap, opts[0]).startswith("Activate Brain Rotter")
    assert "Choose One: BG31_880t2" in enc.describe_option(snap, opts[4])


# --- legal_options parity fixes -------------------------------------------------------
def test_legal_options_skips_cards_flagged_unplayable_and_unaffordable_spells():
    hand = [_m("Lockbox", 0, ctype="SPELL", cid="LB", LITERALLY_UNPLAYABLE=1),
            _m("Stuck", 1, cid="ST", LITERALLY_UNPLAYABLE=1),
            _m("Pricey", 2, ctype="SPELL", cid="PR", COST=4),
            _m("Free", 3, ctype="SPELL", cid="FR", COST=0),
            _m("Minion", 4, cid="MN")]
    plays = _of(enc.legal_options(_snap(hand=hand, gold=3)), "play")
    assert sorted({p["card_id"] for p in plays}) == ["FR", "MN"]
    assert "PR" in {p["card_id"] for p in _of(enc.legal_options(_snap(hand=hand, gold=4)), "play")}


def test_legal_options_activate_from_board():
    rotter = _m("Brain Rotter", 0, cid="BG36_099", eid=7, INTERACTABLE_OBJECT=1)
    rotter_off = _m("Brain Rotter", 1, cid="BG36_099", eid=8, INTERACTABLE_OBJECT=0)
    priced = _m("Priced", 2, cid="BG36_356", eid=9, INTERACTABLE_OBJECT=1,
                INTERACTABLE_OBJECT_COST=2)
    listed = _m("Listed", 3, cid="BG36_180", eid=10)             # no raw tag: use the list
    act = [{"card_id": "BG36_180", "entity_id": 10, "cost": 1, "usable": True}]
    acts = _of(enc.legal_options(_snap(board=[rotter, rotter_off, priced, listed],
                                       activatable=act, gold=1)), "play", "board")
    assert [(o["source"]["slot"], o["card_id"]) for o in acts] == [(0, "BG36_099"), (3, "BG36_180")]
    assert all(o["target"]["zone"] == "none" and o["position"] is None for o in acts)
    acts = _of(enc.legal_options(_snap(board=[priced], gold=2)), "play", "board")
    assert len(acts) == 1 and enc.option_cost(_snap(board=[priced], gold=2), acts[0]) == 2
    act[0]["usable"] = False
    assert not _of(enc.legal_options(_snap(board=[listed], activatable=act)), "play", "board")


def test_legal_options_dark_gift_is_play_with_source_none():
    dg = {"name": "Dark Discovery", "card_id": "BG36_Button_DarkGift", "cost": 3, "usable": True}
    opts = _of(enc.legal_options(_snap(dark_gift=dg)), "play", "none")
    assert opts == [enc.make_option("play", "BG36_Button_DarkGift")]
    assert not _of(enc.legal_options(_snap(dark_gift=dict(dg, usable=False))), "play", "none")
    assert not _of(enc.legal_options(_snap(dark_gift=None)), "play", "none")


# --- trainer: row options, merge, split, pilot -------------------------------------------
def _row(game, d, snap, options, chosen, **extra):
    r = {"state": snap, "game_id": game, "build": BUILD, "turn": 3 + d, "dp_index": d,
         "mmr": 6000 + d, "options": options, "chosen": chosen, "weight": 1.0}
    r.update(extra)
    return r


def _server_rows(n_games, per_game=6, ts=True):
    """Rows whose option lists include options legal_options can't produce
    (targeted spells, hero-targeted duplicates); the expert always plays the
    spell on shop slot 0, which only exists in the server list."""
    rows = []
    for g in range(n_games):
        for d in range(per_game):
            snap = _snap(turn=3 + d, gold=3 + d % 4,
                         board=[_m("b", 0), _m("c", 1)], shop=[_m("x", 0), _m("y", 1)],
                         hand=[_m("Spell", 0, ctype="SPELL", cid="SP")])
            opts = [enc.make_option("end_turn"), enc.make_option("reroll"),
                    enc.make_option("play", "SP", ("hand", 0), target=("board", 0)),
                    enc.make_option("play", "SP", ("hand", 0), target=("board", 1)),
                    enc.make_option("play", "SP", ("hand", 0), target=("shop", 0)),
                    dict(enc.make_option("hero_power", "HP"), target_entity=50),
                    dict(enc.make_option("hero_power", "HP"), target_entity=97)]
            extra = {"created_ts": 1_000_000.0 - 1000 * g + d} if ts else {}
            rows.append(_row(f"g{g:03d}", d, snap, opts, 4, **extra))
    return rows


def test_encode_row_uses_row_options_and_merges_identical_encodings():
    from ml import train_bc_policy as T
    row = _server_rows(1)[0]
    s, o, c = T.encode_row(row)
    assert s.shape == (enc.STATE_DIM,)
    assert o.shape == (6, enc.OPTION_DIM)          # two hero-targeted hero powers merged
    assert c == 4
    assert np.array_equal(o[c], enc.encode_option(row["state"], row["options"][4]))
    ours = {enc.option_key(x) for x in enc.legal_options(row["state"])}
    assert enc.option_key(row["options"][4]) not in ours     # only the server lists it


def test_heldout_orders_games_by_created_ts_then_game_id():
    from ml import train_bc_policy as T
    rows = _server_rows(40, per_game=1)            # g000 is the LATEST by created_ts
    ids, rule = T.make_heldout(rows, BUILD)
    assert "created_ts" in rule and len(ids) == 30
    assert "g000" in ids and "g039" not in ids
    no_ts = _server_rows(40, per_game=1, ts=False)
    ids, rule = T.make_heldout(no_ts, BUILD)
    assert "game_id" in rule and ids == [f"g{g:03d}" for g in range(10, 40)]
    iso = [dict(r, created_at="2026-09-%02dT12:00:00Z" % (28 - int(r["game_id"][1:]) % 28))
           for r in no_ts]
    assert T.game_order(iso, BUILD)[1] == "created_ts"
    with pytest.raises(ValueError):
        T.make_heldout(_server_rows(5, per_game=1), BUILD)
    ids, rule = T.make_heldout(_server_rows(5, per_game=1), BUILD, pilot=True)
    assert ids == ["g000"] and rule.startswith("pilot")


def test_pilot_mode_trains_on_row_options_and_never_touches_real_files(tmp_path, monkeypatch):
    from ml import train_bc_policy as T
    real_heldout = tmp_path / "real" / "policy_heldout_ids.json"
    live = tmp_path / "ml" / "policy_net.pt"
    monkeypatch.setattr(T, "HELDOUT_PATH", str(real_heldout))
    monkeypatch.setattr(T, "LIVE_PATH", str(live))
    data = tmp_path / "rows.jsonl.gz"
    with gzip.open(data, "wt", encoding="utf-8") as fh:
        for r in _server_rows(5):
            fh.write(json.dumps(r) + "\n")
    pilot = tmp_path / "pilot"
    assert T.main(["--data", str(data), "--make-heldout"]) == 2     # too few games
    assert not real_heldout.exists()
    assert T.main(["--data", str(data), "--pilot", "--install", "--pilot-dir", str(pilot)]) == 2
    assert T.main(["--data", str(data), "--pilot", "--pilot-dir", str(pilot),
                   "--heldout", str(real_heldout)]) == 2
    rc = T.main(["--data", str(data), "--pilot", "--pilot-dir", str(pilot),
                 "--epochs", "30", "--lr", "3e-3"])
    assert rc == 0 and not real_heldout.exists() and not live.exists()
    m = json.loads((pilot / "pilot_policy.metrics.json").read_text(encoding="utf-8"))
    assert m["label"].startswith("PLUMBING CHECK")
    assert m["split"]["heldout_game_ids"] == ["g000"] and m["split"]["train_games"] == 4
    assert m["split"]["merged_identical_options"] == {"train": 24, "heldout": 6}
    assert m["decisions"]["heldout"]["by_kind"] == {"play:hand": 6}
    model, rnd = m["heldout"]["model"]["overall"], m["heldout"]["random"]["overall"]
    assert model["n"] == 6 and rnd["top1"] == pytest.approx(1 / 6, abs=1e-6)
    assert model["top1"] == 1.0                     # learned the server-only option
    assert "play:hand" in m["heldout"]["model"]["per_kind"]


# --- parity script -------------------------------------------------------------
def test_parity_script_counts(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "legal_options_parity", REPO / "scripts" / "legal_options_parity.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    snap = _snap(gold=0, board=[_m("b", 0)], hand=[_m("Spell", 0, ctype="SPELL", cid="SP")])
    ours = enc.legal_options(snap)
    exact = _row("g1", 0, snap, list(ours), 0)
    targeted = [o for o in ours if o["type"] != "play"] + [
        enc.make_option("play", "SP", ("hand", 0), target=("board", 0))]
    chosen_t = len(targeted) - 1
    rows = [exact, _row("g1", 1, snap, targeted, chosen_t),
            _row("g1", 2, snap, [enc.make_option("end_turn"),
                                 enc.make_option("reroll")], 1),
            {"game_id": "bad", "options": [], "chosen": 0}]
    r = P.parity(rows)
    assert r["rows"] == 3 and r["rows_skipped_invalid"] == 1
    assert r["exact_match"] == 1 and r["match_ignoring_targets"] == 2
    assert r["chosen_producible"] == 1 and r["chosen_producible_ignoring_targets"] == 2
    assert r["missing_by_type"] == {"play": 1, "reroll": 1}
    assert r["unproducible_chosen_by_kind"] == {"play:spell:targeted": 1, "reroll": 1}
    assert r["extra_by_type"]["play"] == 2 and r["extra_by_type"]["sell"] == 1
    data = tmp_path / "rows.jsonl"
    data.write_text("\n".join(json.dumps(x) for x in rows[:3]) + "\n", encoding="utf-8")
    out = tmp_path / "parity.json"
    assert P.main(["--data", str(data), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["exact_match"] == 1
