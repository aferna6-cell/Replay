"""Behaviour-cloned NEXT policy: shared-encoder parity, legal options, the
training split and metrics, the live HSBG_NEXT_POLICY wiring, and the smoke
script."""

import json
import os
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from hsbg_coach import encode as enc  # noqa: E402
from hsbg_coach.bg import BGTracker, MinionView, Snapshot  # noqa: E402
from hsbg_coach.parser import parse_line  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
POWER_LOG = REPO / "Power.log"
BUILD = 253216


def _m(name, i, atk=2, hp=2, ctype="MINION", cid=None):
    return {"entity_id": 100 + i, "card_id": cid or f"TEST_{i}", "name": name,
            "attack": atk, "health": hp, "position": i + 1, "tags": {"CARDTYPE": ctype}}


def _snap(**kw):
    d = {"game_counter": 1, "turn": 5, "phase": "recruit", "tavern_tier": 2, "gold": 10,
         "hero_health": 30, "board": [], "shop": [], "shop_spells": [], "hand_spells": [],
         "hand": [], "hero_power": None, "level_cost": None}
    d.update(kw)
    return d


def _types(opts):
    return [o["type"] for o in opts]


def _of(opts, t):
    return [o for o in opts if o["type"] == t]


# --- encoder parity ---------------------------------------------------------
@pytest.fixture(scope="module")
def log_snapshot():
    if not POWER_LOG.is_file():
        pytest.skip("repo Power.log sample not present")
    t = BGTracker()
    with POWER_LOG.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            ev = parse_line(line)
            if ev:
                t.feed(ev)
    snap = t.snapshot()
    assert snap.phase == "recruit" and snap.board and snap.shop
    return snap


def _rebuild(snap):
    """Snapshot from its dict form: #98's from_dict when present, else asdict."""
    if hasattr(Snapshot, "from_dict"):
        return Snapshot.from_dict(snap.to_dict())
    d = asdict(snap)
    for zone in ("board", "shop", "hand"):
        d[zone] = [MinionView(**m) for m in d[zone]]
    return Snapshot(**d)


def test_encoder_parity_live_snapshot_vs_dict_forms(log_snapshot):
    ref_state = enc.encode_state(log_snapshot)
    ref_opts = enc.legal_options(log_snapshot)
    assert ref_state.shape == (enc.STATE_DIM,) and np.isfinite(ref_state).all()
    assert ref_opts and all(set(o) == set(enc.OPTION_KEYS) for o in ref_opts)
    ref_ov = np.stack([enc.encode_option(log_snapshot, o) for o in ref_opts])
    assert ref_ov.shape == (len(ref_opts), enc.OPTION_DIM)
    forms = [log_snapshot.to_dict(), _rebuild(log_snapshot),
             json.loads(json.dumps(log_snapshot.to_dict()))]   # a jsonl training row
    for form in forms:
        assert np.array_equal(enc.encode_state(form), ref_state)
        opts = enc.legal_options(form)
        assert opts == ref_opts
        assert np.array_equal(np.stack([enc.encode_option(form, o) for o in opts]), ref_ov)


def test_encoder_ignores_debug_fields_and_reads_optional_fields(log_snapshot):
    d = log_snapshot.to_dict()
    opt = enc.legal_options(d)[0]
    noisy = dict(opt, src_entity=123, target_entity=456)
    assert np.array_equal(enc.encode_option(d, opt), enc.encode_option(d, noisy))
    assert enc.option_key(opt) == enc.option_key(noisy)
    extra = dict(d, hero_armor=5, shop_frozen=True)
    assert not np.array_equal(enc.encode_state(extra), enc.encode_state(d))


# --- legal_options ----------------------------------------------------------
def test_no_gold_means_no_buy_roll_level_or_hero_power():
    s = _snap(gold=0, tavern_tier=1, board=[_m("a", 0), _m("b", 1)],
              shop=[_m("x", 0), _m("y", 1), _m("z", 2)],
              hero_power={"name": "HP", "card_id": "HP1", "cost": 1, "usable": True})
    types = _types(enc.legal_options(s))
    for t in ("buy", "reroll", "level", "hero_power"):
        assert t not in types
    assert types.count("sell") == 2 and "freeze" in types and types[-1] == "end_turn"


def test_buy_needs_three_gold_and_a_free_hand_slot():
    shop = [_m("x", 0), _m("y", 1), _m("z", 2)]
    assert not _of(enc.legal_options(_snap(gold=2, shop=shop)), "buy")
    buys = _of(enc.legal_options(_snap(gold=3, shop=shop)), "buy")
    assert [b["source"] for b in buys] == [{"zone": "shop", "slot": i} for i in range(3)]
    assert [b["card_id"] for b in buys] == ["TEST_0", "TEST_1", "TEST_2"]
    full_hand = [_m(f"h{i}", i) for i in range(10)]
    assert not _of(enc.legal_options(_snap(gold=10, shop=shop, hand=full_hand)), "buy")
    full_board = [_m(f"b{i}", i) for i in range(7)]          # buying goes to hand
    assert len(_of(enc.legal_options(_snap(gold=3, shop=shop, board=full_board)), "buy")) == 3


def test_shop_spell_uses_its_own_cost_and_slots_after_minions():
    shop = [_m("x", 0), _m("y", 1)]
    spells = [{"name": "Spell", "card_id": "SP1", "cost": 4}]
    cheap = _of(enc.legal_options(_snap(gold=3, shop=shop, shop_spells=spells)), "buy")
    assert [b["card_id"] for b in cheap] == ["TEST_0", "TEST_1"]
    snap = _snap(gold=4, shop=shop, shop_spells=spells)
    sp = [b for b in _of(enc.legal_options(snap), "buy") if b["card_id"] == "SP1"]
    assert sp and sp[0]["source"] == {"zone": "shop", "slot": 2}
    assert enc.describe_option(snap, sp[0]) == "Buy spell: Spell (4g)"


def test_play_respects_board_space():
    hand = [_m("minion", 0, cid="HM"), _m("spell", 1, ctype="BATTLEGROUND_SPELL", cid="HS"),
            _m("chrome", 2, cid="TB_BaconShop_DragBuy")]
    board3 = [_m(f"b{i}", i) for i in range(3)]
    plays = _of(enc.legal_options(_snap(board=board3, hand=hand)), "play")
    assert [p["position"] for p in plays if p["card_id"] == "HM"] == [0, 1, 2, 3]
    assert [p["position"] for p in plays if p["card_id"] == "HS"] == [None]
    assert all(p["source"]["zone"] == "hand" for p in plays)
    assert not [p for p in plays if "DragBuy" in (p["card_id"] or "")]
    board7 = [_m(f"b{i}", i) for i in range(7)]
    full = _of(enc.legal_options(_snap(board=board7, hand=hand)), "play")
    assert [p["card_id"] for p in full] == ["HS"]           # no minion onto a full board


def test_level_only_when_affordable_and_below_tier_six():
    def can_level(**kw):
        return "level" in _types(enc.legal_options(_snap(**kw)))
    assert not can_level(tavern_tier=3, level_cost=6, gold=5)
    assert can_level(tavern_tier=3, level_cost=6, gold=6)
    assert can_level(tavern_tier=1, level_cost=None, gold=5)     # base cost 5
    assert not can_level(tavern_tier=1, level_cost=None, gold=4)
    assert not can_level(tavern_tier=6, level_cost=None, gold=99)


def test_hero_power_needs_usable_and_affordable():
    hp = {"name": "HP", "card_id": "HP1", "cost": 2, "usable": True}
    assert not _of(enc.legal_options(_snap(gold=1, hero_power=hp)), "hero_power")
    assert _of(enc.legal_options(_snap(gold=2, hero_power=hp)), "hero_power")[0]["card_id"] == "HP1"
    assert not _of(enc.legal_options(_snap(gold=9, hero_power=dict(hp, usable=False))), "hero_power")
    assert not _of(enc.legal_options(_snap(gold=9, hero_power=None)), "hero_power")


def test_reposition_pairs_and_non_recruit_phase():
    rep = _of(enc.legal_options(_snap(board=[_m(f"b{i}", i) for i in range(3)])), "reposition")
    assert len(rep) == 6 and all(r["position"] != r["source"]["slot"] for r in rep)
    assert not _of(enc.legal_options(_snap(board=[_m("b", 0)])), "reposition")
    assert enc.legal_options(_snap(phase="combat", shop=[_m("x", 0)])) == []


# --- training ---------------------------------------------------------------
def _synthetic_rows(n_games=80, per_game=4, seed=0):
    """Expert rule: buy the highest-attack shop minion. Ten other-build games
    always end turn (must be ignored); one forced row per game is skipped."""
    from hsbg_coach.synergy import load_embeddings
    rng = random.Random(seed)
    names = sorted(load_embeddings())[:30] or [f"card{i}" for i in range(30)]

    def decision(game, build, d, expert=True):
        n_shop = rng.randint(3, 5)
        atks = rng.sample(range(1, 13), n_shop)
        shop = [_m(rng.choice(names), i, atk=atks[i], hp=rng.randint(1, 8), cid=f"S{i}")
                for i in range(n_shop)]
        board = [_m(rng.choice(names), i, atk=rng.randint(1, 8), hp=rng.randint(1, 8),
                    cid=f"B{i}") for i in range(rng.randint(1, 4))]
        snap = _snap(turn=3 + d, gold=rng.randint(3, 10), tavern_tier=rng.randint(1, 4),
                     board=board, shop=shop)
        opts = enc.legal_options(snap)
        if expert:
            best = atks.index(max(atks))
            chosen = next(k for k, o in enumerate(opts)
                          if o["type"] == "buy" and o["source"]["slot"] == best)
        else:
            chosen = len(opts) - 1                             # end_turn
        return {"snapshot": snap, "game_id": game, "build": build, "turn": 3 + d,
                "dp_index": d, "mmr": 5000 + 10 * rng.randint(0, 300),
                "options": opts, "chosen": chosen, "weight": 1.0}

    rows = []
    for g in range(10):
        rows += [decision(f"old{g:03d}", 250000, d, expert=False) for d in range(per_game)]
    for g in range(n_games):
        gid = f"g{g:03d}"
        rows += [decision(gid, BUILD, d) for d in range(per_game)]
        rows.append({"snapshot": _snap(), "game_id": gid, "build": BUILD, "turn": 9,
                     "dp_index": per_game, "mmr": 6000,
                     "options": [enc.make_option("end_turn")], "chosen": 0, "weight": 1.0})
    return rows


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from ml import train_bc_policy as T
    d = tmp_path_factory.mktemp("bc_policy")
    rows = _synthetic_rows()
    data = d / "rows.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    heldout, out = d / "heldout.json", d / "policy.pt"
    rc = T.main(["--data", str(data), "--heldout", str(heldout), "--make-heldout",
                 "--epochs", "25", "--lr", "3e-3", "--out", str(out)])
    assert rc == 0
    metrics = json.loads((d / "policy.metrics.json").read_text(encoding="utf-8"))
    return {"T": T, "dir": d, "rows": rows, "data": data, "heldout": heldout,
            "out": out, "metrics": metrics}


def test_heldout_split_is_recent_games_and_never_trained(trained):
    T = trained["T"]
    held = T.read_heldout(str(trained["heldout"]))
    assert held == {f"g{g:03d}" for g in range(50, 80)}      # 15% of 80 < 30 -> 30 newest
    train_rows, held_rows, counts = T.split_rows(trained["rows"], held, BUILD)
    assert not {r["game_id"] for r in train_rows} & held
    assert {r["game_id"] for r in held_rows} <= held
    assert all(r["build"] == BUILD for r in train_rows)      # other build never trains
    assert counts["other_build"] == 40
    assert counts["forced_train"] == 50 and counts["forced_heldout"] == 30
    split = trained["metrics"]["split"]
    assert split["overlap"] == 0 and split["train_rows"] == 200 and split["heldout_rows"] == 120


def test_training_beats_random_on_heldout(trained):
    h = trained["metrics"]["heldout"]
    model, rnd = h["model"]["overall"], h["random"]["overall"]
    assert model["n"] == 120
    assert model["top1"] > rnd["top1"] + 0.25, (model, rnd)
    assert model["top3"] >= model["top1"]
    assert "buy" in h["model"]["per_type"]
    assert trained["metrics"]["state_dim"] == enc.STATE_DIM
    assert trained["metrics"]["option_dim"] == enc.OPTION_DIM


def test_training_is_deterministic_and_compare_scores_same_rows(trained):
    T = trained["T"]
    out2 = trained["dir"] / "policy2.pt"
    rc = T.main(["--data", str(trained["data"]), "--heldout", str(trained["heldout"]),
                 "--epochs", "25", "--lr", "3e-3", "--out", str(out2),
                 "--compare", str(trained["out"])])
    assert rc == 0
    a = torch.load(trained["out"], weights_only=True)["state_dict"]
    b = torch.load(out2, weights_only=True)["state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    m2 = json.loads((trained["dir"] / "policy2.metrics.json").read_text(encoding="utf-8"))
    assert m2["heldout"]["compare"] == trained["metrics"]["heldout"]["model"]


def test_mismatched_checkpoint_refuses_to_load(trained, tmp_path):
    from ml.bc_policy import CheckpointMismatch, load_bc_policy
    assert load_bc_policy(str(trained["out"])).meta["encoder_version"] == enc.ENCODER_VERSION
    ckpt = torch.load(trained["out"], weights_only=True)
    for key, bad in (("encoder_version", "bc-enc-v0"),
                     ("encoder_version", "bc-enc-v4"),
                     ("state_dim", enc.STATE_DIM + 1)):
        broken = tmp_path / f"{key}-{bad}.pt"
        torch.save({"meta": dict(ckpt["meta"], **{key: bad}),
                    "state_dict": ckpt["state_dict"]}, broken)
        with pytest.raises(CheckpointMismatch):
            load_bc_policy(str(broken))


# --- live wiring ------------------------------------------------------------
def test_next_policy_flag_and_checkpoint_fallbacks(monkeypatch, tmp_path, trained, log_snapshot, capsys):
    from hsbg_coach import live
    from hsbg_coach.overlay import format_next
    monkeypatch.delenv("HSBG_NEXT_POLICY", raising=False)
    monkeypatch.setenv("HSBG_NEXT_POLICY_PATH", str(trained["out"]))
    assert live._load_next_policy() is not None              # unset means on
    for off in ("0", "false", "off", "FALSE", "Off"):
        monkeypatch.setenv("HSBG_NEXT_POLICY", off)
        assert live._load_next_policy() is None              # explicit off -> advisor
    monkeypatch.setenv("HSBG_NEXT_POLICY", "1")
    monkeypatch.setenv("HSBG_NEXT_POLICY_PATH", str(tmp_path / "missing.pt"))
    assert live._load_next_policy() is None                  # missing checkpoint -> advisor
    err = capsys.readouterr().err
    assert "falling back to the eval-net advisor" in err
    assert "checkpoint not found" in err
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    monkeypatch.setenv("HSBG_NEXT_POLICY_PATH", str(bad))
    assert live._load_next_policy() is None                  # corrupt checkpoint -> advisor
    err = capsys.readouterr().err
    assert "falling back to the eval-net advisor" in err
    assert "failed to load" in err
    monkeypatch.delenv("HSBG_NEXT_POLICY", raising=False)    # unset + good file still on
    monkeypatch.setenv("HSBG_NEXT_POLICY_PATH", str(trained["out"]))
    policy = live._load_next_policy()
    assert policy is not None

    snap = log_snapshot.to_dict()
    lines = ["Buy Foo (finish 3.1) - why", "Roll the shop"]
    assert live.policy_next_lines(lines, snap, None) is lines   # no policy: unchanged
    new = live.policy_next_lines(lines, snap, policy)
    expected = enc.describe_option(snap, policy.best(snap))
    assert new[0] == expected and new[1:] == [l for l in lines if l != expected]
    assert expected in format_next(snap, None, new).splitlines()[0]


def test_livecoach_policy_defaults_on_unless_off_or_checkpoint_missing(monkeypatch, trained, capsys):
    from hsbg_coach.live import LiveCoach
    monkeypatch.delenv("HSBG_NEXT_POLICY", raising=False)
    monkeypatch.setenv("HSBG_NEXT_POLICY_PATH", str(trained["out"]))
    assert LiveCoach(power_log=None)._next_policy is not None   # unset means on
    monkeypatch.setenv("HSBG_NEXT_POLICY", "0")
    assert LiveCoach(power_log=None)._next_policy is None       # 0 means advisor
    monkeypatch.delenv("HSBG_NEXT_POLICY", raising=False)
    monkeypatch.setenv("HSBG_NEXT_POLICY_PATH", str(trained["dir"] / "nope.pt"))
    assert LiveCoach(power_log=None)._next_policy is None       # missing -> advisor
    err = capsys.readouterr().err
    assert "falling back to the eval-net advisor" in err
    monkeypatch.setenv("HSBG_NEXT_POLICY_PATH", str(trained["out"]))
    assert LiveCoach(power_log=None)._next_policy is not None


def _run_smoke(checkpoint):
    env = {k: v for k, v in os.environ.items() if not k.startswith("HSBG_NEXT_POLICY")}
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "policy_smoke.py"), "--checkpoint", str(checkpoint)],
        cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=900)


def test_smoke_script_on_fresh_checkpoint(trained):
    if not POWER_LOG.is_file():
        pytest.skip("repo Power.log sample not present")
    ok = _run_smoke(trained["out"])
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "OK: NEXT from policy" in ok.stdout
    missing = _run_smoke(trained["dir"] / "missing.pt")
    assert missing.returncode != 0