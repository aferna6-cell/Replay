"""VOD reconstruction: segmentation, action inference, recorder-schema parity,
the validate gate, and end-to-end ingestion into trajectory_examples."""

import json

import pytest

from vod.reconstruct import FrameRead, infer_actions, reconstruct
from vod import validate


def _read(ts, turn, board, gold=5, tier=2, hp=30, conf=0.9, phase="recruit",
          placement=None):
    return FrameRead(ts=ts, phase=phase, state={
        "phase": phase, "turn": turn, "tavern_tier": tier, "gold": gold,
        "hero_health": hp, "hero_name": "Reno Jackson",
        "board": [{"name": n, "attack": 2, "health": 2, "golden": False}
                  for n in board],
        "shop": [], "final_placement": placement, "confidence": conf,
    })


def test_reconstruct_single_game_last_read_per_turn_wins():
    reads = [
        _read(0, 1, ["Alleycat"]),
        _read(3, 1, ["Alleycat", "Sellemental"]),      # later read, same turn
        FrameRead(ts=6, phase="combat"),
        _read(9, 2, ["Alleycat", "Sellemental", "Manasaber"]),
        FrameRead(ts=12, phase="endscreen",
                  state=_read(12, None, [], phase="endscreen",
                              placement=2).state),
    ]
    games = reconstruct(reads, "abc123")
    assert len(games) == 1
    records = games[0]
    assert len(records) == 2
    assert records[0]["game_id"] == "vod-abc123-g1"
    assert records[0]["placement"] == 2
    # turn 1 kept the LAST confident read
    names = [m["name"] for m in records[0]["state"]["board"]]
    assert names == ["Alleycat", "Sellemental"]
    assert records[0]["source"] == "vod"


def test_reconstruct_splits_games_on_turn_reset():
    reads = [_read(0, 5, ["Alleycat"]), _read(3, 6, ["Alleycat"]),
             _read(10, 1, ["Sellemental"]), _read(13, 2, ["Sellemental"])]
    games = reconstruct(reads, "v")
    assert len(games) == 2
    assert games[0][0]["game_id"] == "vod-v-g1"
    assert games[1][0]["game_id"] == "vod-v-g2"
    assert games[1][0]["placement"] is None       # no endscreen read


def test_low_confidence_reads_dropped():
    games = reconstruct([_read(0, 1, ["Alleycat"], conf=0.2)], "v")
    assert games == []


def test_infer_actions_diff():
    prev = _read(0, 3, ["Alleycat", "Alleycat", "Manasaber"], tier=2).state
    prev = reconstruct([_read(0, 3, ["Alleycat", "Alleycat", "Manasaber"],
                              tier=2)], "x")[0][0]["state"]
    cur = reconstruct([_read(0, 4, ["Alleycat", "Sellemental"],
                             tier=3)], "x")[0][0]["state"]
    acts = infer_actions(prev, cur)
    kinds = {(a["type"], a.get("name")) for a in acts}
    assert ("tier_up", None) in {(a["type"], None if a["type"] == "tier_up"
                                  else a.get("name")) for a in acts}
    assert ("play", "Sellemental") in kinds
    assert ("sell", "Alleycat") in kinds
    assert ("sell", "Manasaber") in kinds


def test_records_feed_trajectory_examples(tmp_path):
    np = pytest.importorskip("numpy")  # noqa: F841 — board features need it
    from ml.board_dataset import trajectory_examples, example_weights
    reads = [_read(0, 1, ["Alleycat", "Sellemental"]),
             FrameRead(ts=3, phase="endscreen",
                       state=_read(3, None, [], phase="endscreen",
                                   placement=4).state)]
    records = reconstruct(reads, "e2e")[0]
    path = tmp_path / "vod-e2e-g1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    ex = trajectory_examples(str(tmp_path), weight=3.0)
    assert len(ex) == 1
    assert ex[0]["label"] == 4.0
    assert ex[0]["weight"] == 3.0
    assert list(example_weights(ex)) == [3.0]


def test_validate_compare():
    truth = [{"state": {"turn": 1, "gold": 3, "tavern_tier": 1,
                        "hero_health": 30,
                        "board": [{"name": "Alleycat"}]}, "placement": 1}]
    vod_good = [{"state": {"turn": 1, "gold": 3, "tavern_tier": 1,
                           "hero_health": 30,
                           "board": [{"name": "Alleycat"}]}}]
    rep = validate.compare(vod_good, truth)
    assert rep["board_accuracy"] == 1.0
    assert rep["exact"]["gold"] == 1.0
    vod_bad = [{"state": {"turn": 1, "gold": 9, "tavern_tier": 1,
                          "hero_health": 30,
                          "board": [{"name": "Wrong Minion"}]}}]
    rep = validate.compare(vod_bad, truth)
    assert rep["board_accuracy"] == 0.0
    assert rep["exact"]["gold"] == 0.0


def test_weighted_training_runs():
    torch = pytest.importorskip("torch")  # noqa: F841
    np = pytest.importorskip("numpy")
    from ml.eval_net import train
    X = np.random.RandomState(0).rand(40, 6).astype(np.float32)
    y = np.random.RandomState(1).rand(40).astype(np.float32) * 7 + 1
    h = np.zeros(40, dtype=np.int64)
    w = np.ones(40, dtype=np.float32)
    w[:10] = 3.0
    model, _ = train(X, h, y, n_heroes=1, epochs=2, verbose=False,
                     sample_weight=w)
    assert model is not None
