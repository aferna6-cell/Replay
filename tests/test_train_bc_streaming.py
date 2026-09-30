"""The streaming trainer (one file at a time, option matrices memmapped on disk)
gives the same weights and metrics as the old all-in-memory path; frozen
held-out ids; the validation run; the gate."""

import gzip
import hashlib
import json
import random

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from hsbg_coach import encode as enc  # noqa: E402

BUILD = 253216
O = enc.make_option


def _m(name, i, cid=None, atk=2, hp=2, **tags):
    return {"entity_id": 100 + i, "card_id": cid or f"C{i}", "name": name, "attack": atk,
            "health": hp, "position": i + 1,
            "tags": dict({"CARDTYPE": "MINION"}, **{k: str(v) for k, v in tags.items()})}


def _rows(n_games=36, per_game=5, seed=3):
    """Synthetic label rows: the expert buys the cheapest shop slot, else rolls.
    One old-build game, forced single-option rows, merged duplicate options."""
    rng = random.Random(seed)
    games = []
    for g in range(n_games + 1):
        build = BUILD if g < n_games else 250000
        rows = []
        for d in range(per_game):
            shop = [dict(_m("s%d" % i, i, cid="S%d" % rng.randint(0, 5), atk=rng.randint(1, 6)),
                         buy_cost=rng.choice([1, 2, 3])) for i in range(3)]
            board = [_m("b%d" % i, i, atk=rng.randint(1, 6)) for i in range(rng.randint(1, 4))]
            snap = {"turn": 3 + d, "phase": "recruit", "tavern_tier": 2, "gold": rng.randint(2, 6),
                    "hero_health": 30, "board": board, "shop": shop, "hand": [],
                    "shop_spells": [], "reroll_cost": rng.choice([0, 1]), "level_cost": 5,
                    "hero_power": {"name": "HP", "card_id": "HPX", "cost": 1, "usable": True}}
            opts = [O("end_turn"), O("reroll"), O("freeze")]
            opts += [O("buy", m["card_id"], ("shop", i)) for i, m in enumerate(shop)]
            opts += [O("sell", m["card_id"], ("board", i)) for i, m in enumerate(board)]
            opts += [dict(O("hero_power", "HPX"), target_entity=t) for t in (1, 2)]
            if len(board) >= 2:
                opts += [O("reposition", board[0]["card_id"], ("board", 0), position=1)]
            cheapest = min(range(3), key=lambda i: shop[i]["buy_cost"])
            chosen = 3 + cheapest if shop[cheapest]["buy_cost"] <= snap["gold"] else 1
            rows.append({"state": snap, "game_id": "g%03d" % g, "build": build,
                         "dp_index": d, "turn": 3 + d, "mmr": 4000 + 37 * g,
                         "options": opts, "chosen": chosen, "weight": 0.2 + 0.02 * g,
                         "created_ts": 1_790_000_000_000 + 60_000 * g + d})
        rows.append({"state": dict(snap), "game_id": "g%03d" % g, "build": build,
                     "dp_index": per_game, "turn": 9, "mmr": 4000 + 37 * g,
                     "options": [O("end_turn")], "chosen": 0, "weight": 0.5,
                     "created_ts": 1_790_000_000_000 + 60_000 * g + per_game})
        games.append(rows)
    return games


def _write(tmp_path, games):
    d = tmp_path / "labels"
    d.mkdir()
    for rows in games:
        with gzip.open(d / f"{rows[0]['game_id']}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return str(d / "*.jsonl.gz")


def _in_memory(T, pattern, heldout, epochs, lr, batch, scorers_extra=None):
    """The pre-streaming main(): every row in memory, same building blocks."""
    from ml.bc_policy import BCPolicy
    rows = T.read_rows([pattern])
    train_rows, held_rows, counts = T.split_rows(rows, heldout, BUILD)
    items = [T.encode_row(r) + (T.row_weight(r, BUILD),) for r in train_rows]
    model, history = T.train_model(items, epochs, lr, batch, 0)
    held_enc = [T.encode_row(r) for r in held_rows]
    groups = [T.encode_row(r, with_groups=True)[3] for r in held_rows]
    results = T.evaluate(held_rows, held_enc, {"model": BCPolicy(model).score_encoded},
                         extra=scorers_extra, groups=groups)
    return model, history, results, counts, held_rows


def _load(path):
    return torch.load(path, weights_only=True)["state_dict"]


@pytest.mark.parametrize("workers", [1, 2])
def test_streaming_matches_in_memory_path(tmp_path, workers):
    from ml import train_bc_policy as T
    pattern = _write(tmp_path, _rows())
    heldout = tmp_path / "heldout.json"
    out = tmp_path / "policy.pt"
    rc = T.main(["--data", pattern, "--heldout", str(heldout), "--make-heldout",
                 "--epochs", "4", "--lr", "3e-3", "--batch", "16", "--out", str(out),
                 "--no-baseline-evalnet", "--workers", str(workers), "--train-fit"])
    assert rc == 0
    m = json.loads((tmp_path / "policy.metrics.json").read_text(encoding="utf-8"))
    ids = set(T.read_heldout(str(heldout)))
    assert ids == set(T.make_heldout(T.read_rows([pattern]), BUILD)[0]) and len(ids) == 30
    model, history, results, counts, _ = _in_memory(T, pattern, ids, 4, 3e-3, 16)
    got = _load(out)
    assert all(torch.equal(got[k], model.state_dict()[k]) for k in got)
    assert m["train_loss"] == history
    for name in ("model", "model_coarse", "random"):
        assert m["heldout"][name] == results[name]
    for k in ("rows", "other_build", "forced_train", "forced_heldout", "invalid"):
        assert m["split"][k] == counts[k]
    assert m["split"]["merged_identical_options"]["heldout"] > 0
    assert m["train_fit"]["model"]["overall"]["n"] == m["split"]["train_rows"]
    assert m["heldout_frozen"] is False
    assert m["heldout_file_sha256"] == hashlib.sha256(heldout.read_bytes()).hexdigest()
    assert not list(tmp_path.glob("**/*_options.f32"))          # cache removed
    assert not list(tmp_path.glob("**/*_states.f32"))


def test_streaming_advisor_matches_run_advisor_and_gate(tmp_path, monkeypatch):
    from hsbg_coach.board_value import HeuristicScorer
    from ml import advisor_baseline as AB
    from ml import train_bc_policy as T
    monkeypatch.setattr(AB, "load_scorer", lambda path=None: HeuristicScorer())
    pattern = _write(tmp_path, _rows(n_games=34, per_game=3))
    heldout, out = tmp_path / "heldout.json", tmp_path / "p.pt"
    rc = T.main(["--data", pattern, "--heldout", str(heldout), "--make-heldout",
                 "--epochs", "2", "--batch", "16", "--out", str(out),
                 "--compare", "advisor", "--advisor-both"])
    assert rc == 0
    m = json.loads((tmp_path / "p.metrics.json").read_text(encoding="utf-8"))
    ids = T.read_heldout(str(heldout))
    rows = T.read_rows([pattern])
    _, held_rows, _ = T.split_rows(rows, ids, BUILD)
    plain = T.run_advisor(held_rows, hand_lines=False)
    hand = T.run_advisor(held_rows, hand_lines=True)
    _, _, results, _, _ = _in_memory(T, pattern, ids, 2, 1e-3, 16,
                                     {"advisor": plain["hits"], "compare": plain["hits"],
                                      "advisor_hand_lines": hand["hits"]})
    for name in ("advisor", "advisor_hand_lines", "compare", "model", "model_coarse"):
        assert m["heldout"][name] == results[name]
    assert m["advisor_baseline"]["mapping"] == plain["mapping"]
    assert m["advisor_baseline"]["advisor_hand_lines"]["mapping"] == hand["mapping"]
    g = m["gate"]
    assert g["result"] in ("PASS", "FAIL") and set(g["advisors"]) == {"advisor",
                                                                       "advisor_hand_lines"}
    assert g == T.gate(results, ["advisor", "advisor_hand_lines"])


def test_frozen_heldout_ids_used_as_is_and_missing_reported(tmp_path):
    from ml import train_bc_policy as T
    pattern = _write(tmp_path, _rows(n_games=34, per_game=2))
    frozen = tmp_path / "frozen.json"
    ids = ["g000", "g005", "g010", "gone_1"]
    frozen.write_text(json.dumps({"build": str(BUILD), "game_ids": ids}), encoding="utf-8")
    before = frozen.read_bytes()
    out = tmp_path / "f.pt"
    assert T.main(["--data", pattern, "--heldout-ids", str(tmp_path / "nope.json"),
                   "--out", str(out)]) == 2
    rc = T.main(["--data", pattern, "--heldout-ids", str(frozen), "--make-heldout",
                 "--epochs", "1", "--out", str(out), "--no-baseline-evalnet"])
    assert rc == 0 and frozen.read_bytes() == before
    m = json.loads((tmp_path / "f.metrics.json").read_text(encoding="utf-8"))
    assert m["heldout_frozen"] is True
    assert m["split"]["heldout_game_ids"] == sorted(ids)
    assert m["split"]["heldout_ids_missing_from_data"] == ["gone_1"]
    assert m["split"]["heldout_rows"] == 3 * 2              # forced rows excluded
    assert m["heldout_file_sha256"] == hashlib.sha256(before).hexdigest()


def test_validation_run_uses_training_games_only(tmp_path):
    from ml import train_bc_policy as T
    pattern = _write(tmp_path, _rows(n_games=70, per_game=2))
    frozen = tmp_path / "frozen.json"
    held = ["g%03d" % g for g in range(60, 70)]
    frozen.write_text(json.dumps({"game_ids": held}), encoding="utf-8")
    out = tmp_path / "v.pt"
    rc = T.main(["--data", pattern, "--heldout-ids", str(frozen), "--val", "--epochs", "3",
                 "--out", str(out)])
    assert rc == 0
    m = json.loads((tmp_path / "v.metrics.json").read_text(encoding="utf-8"))
    v = m["validation"]
    assert [p["epoch"] for p in v["curve"]] == [1, 2, 3]
    assert v["split"]["val_games"] == 30 and v["split"]["fit_games"] == 30
    assert [p["train_loss"] for p in v["curve"]] == m["train_loss"]
    assert "heldout" not in m and m["label"].startswith("validation")
    assert T.main(["--data", pattern, "--heldout-ids", str(frozen), "--val",
                   "--install"]) == 2


def _gate_results(freeze_mc, hp_mc, sell_mc=0.56):
    """model_coarse vs two advisor columns over three types (n >= 50 each)."""
    def col(t):
        return {"overall": {"n": 300, "top1": t["overall"], "top3": None},
                "per_type": {k: {"n": 100, "top1": v, "top3": None}
                             for k, v in t.items() if k != "overall"}}
    return {"model_coarse": col({"overall": 0.40, "freeze": freeze_mc, "hero_power": hp_mc,
                                 "sell": sell_mc}),
            "advisor": col({"overall": 0.10, "freeze": 0.42, "hero_power": 0.70, "sell": 0.2}),
            "advisor_hand_lines": col({"overall": 0.18, "freeze": 0.18, "hero_power": 0.30,
                                       "sell": 0.55})}


def test_gate_freeze_reported_but_exempt_by_default():
    from ml import train_bc_policy as T
    adv = ["advisor", "advisor_hand_lines"]
    # freeze far below the advisor, hero power fine: PASS, freeze reported only
    g = T.gate(_gate_results(0.13, 0.70), adv)
    assert g["result"] == "PASS" and g["failing_types"] == []
    assert g["exempt"] == ["freeze"] and g["exempt_below_threshold"] == ["freeze"]
    f = g["per_type"]["freeze"]
    assert f["exempt"] is True and f["status"] == "reported_but_exempt"
    assert f["pass"] is False and f["gated"] is False
    assert f["delta"] == pytest.approx(0.13 - 0.42)
    assert "exempt" in g["rule"] and "freeze" in g["rule"]
    # hero power stays a hard gate
    g = T.gate(_gate_results(0.13, 0.60), adv)
    assert g["result"] == "FAIL" and g["failing_types"] == ["hero_power"]
    assert g["per_type"]["hero_power"]["exempt"] is False
    # 3.0-point tolerance still applies to non-exempt types (sell: 0.55 best advisor)
    assert T.gate(_gate_results(0.13, 0.70, sell_mc=0.521), adv)["result"] == "PASS"
    assert T.gate(_gate_results(0.13, 0.70, sell_mc=0.519), adv)["failing_types"] == ["sell"]
    # no exemptions: freeze blocks again (the old rule)
    g = T.gate(_gate_results(0.13, 0.70), adv, exempt=())
    assert g["result"] == "FAIL" and g["failing_types"] == ["freeze"]
    assert g["exempt"] == [] and "status" not in g["per_type"]["freeze"]


def test_gate_exempt_cli_default_and_hp_weight_flag():
    from ml import train_bc_policy as T
    a = T.parse_args(["--data", "x"])
    assert a.gate_exempt == ["freeze"] and a.hp_weight == 1.0
    assert T.parse_args(["--data", "x", "--gate-exempt"]).gate_exempt == []
    assert T.parse_args(["--data", "x", "--hp-weight", "4"]).hp_weight == 4.0
    w = T.type_weights([1.0, 0.5, 2.0], ["hero_power", "hero_power", "buy"],
                       {"hero_power": 4.0})
    assert w == [4.0, 2.0, 2.0]
    assert T.type_weights([1.0], ["buy"], {}) == [1.0]


def test_hp_weight_changes_training_and_is_recorded(tmp_path):
    from ml import train_bc_policy as T
    games = _rows(n_games=34, per_game=3)
    for rows in games:                            # the expert uses the hero power first
        r = rows[0]
        r["chosen"] = next(i for i, o in enumerate(r["options"]) if o["type"] == "hero_power")
    pattern = _write(tmp_path, games)
    heldout = tmp_path / "heldout.json"
    outs = {}
    for w in (1.0, 4.0):
        out = tmp_path / f"p{w}.pt"
        rc = T.main(["--data", pattern, "--heldout", str(heldout), "--make-heldout",
                     "--epochs", "2", "--batch", "16", "--out", str(out),
                     "--hp-weight", str(w), "--no-baseline-evalnet"])
        assert rc == 0
        m = json.loads((tmp_path / f"p{w}.metrics.json").read_text(encoding="utf-8"))
        assert m["hp_weight"] == w
        outs[w] = (_load(out), m)
    a, b = outs[1.0][0], outs[4.0][0]
    assert any(not torch.equal(a[k], b[k]) for k in a)
    assert outs[1.0][1]["heldout"]["model"]["overall"]["n"] == \
        outs[4.0][1]["heldout"]["model"]["overall"]["n"]
