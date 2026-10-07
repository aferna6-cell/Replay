"""Block shuffle, float16 cache, and default-order compatibility for the BC trainer.

Defaults (--shuffle-block 0, --cache-dtype float32) must reproduce the old
full permutation. Block shuffle stays inside contiguous cache blocks and
still visits every row once. float16 caches round-trip one-hots exactly.
"""

import json
import os

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from hsbg_coach import encode as enc  # noqa: E402
from tests.test_train_bc_streaming import BUILD, _rows, _write  # noqa: E402


def _seen_items(n, seen):
    """A sequence whose reads record the index train_model asks for."""

    class Items:
        def __len__(self):
            return n

        def __getitem__(self, i):
            seen.append(int(i))
            state = np.zeros(enc.STATE_DIM, np.float32)
            options = np.zeros((1, enc.OPTION_DIM), np.float32)
            return state, options, 0, 1.0

    return Items()


def _old_orders(n, epochs, seed):
    """The pre-block-shuffle loop: one RandomState.permutation per epoch."""
    rng = np.random.RandomState(seed)
    return [rng.permutation(n) for _ in range(epochs)]


def _block_segments(order, block):
    segments = []
    for idx in order:
        b = int(idx) // block
        if not segments or segments[-1][0] != b:
            segments.append([b, []])
        segments[-1][1].append(int(idx))
    return segments


def test_block_shuffle_is_deterministic_and_covers_every_row():
    from ml import train_bc_policy as T
    # batch does not divide the block, so a batch window spans two blocks.
    n, block, epochs, seed, batch = 100, 30, 3, 7, 16
    rng = np.random.RandomState(seed)
    orders = [T.epoch_order(rng, n, block) for _ in range(epochs)]
    again = [T.epoch_order(np.random.RandomState(seed), n, block) for _ in range(1)]
    # A fresh RandomState repeats epoch 0; the trainer's rng then advances.
    assert np.array_equal(orders[0], again[0])
    assert not np.array_equal(orders[0], orders[1])
    n_blocks = (n + block - 1) // block
    for order in orders:
        assert sorted(int(i) for i in order) == list(range(n))
        segments = _block_segments(order, block)
        assert len(segments) == n_blocks
        assert sorted(b for b, _ in segments) == list(range(n_blocks))
        for b, rows in segments:
            start, stop = b * block, min((b + 1) * block, n)
            assert sorted(rows) == list(range(start, stop))
        # Batches are sliced from the flat order, so a batch can span blocks.
        crossed = [order[s:s + batch] for s in range(0, n, batch)]
        assert any(len({int(i) // block for i in window}) > 1 for window in crossed)

    seen_a, seen_b = [], []
    T.train_model(_seen_items(n, seen_a), epochs, 1e-3, batch, seed, shuffle_block=block)
    T.train_model(_seen_items(n, seen_b), epochs, 1e-3, batch, seed, shuffle_block=block)
    assert seen_a == seen_b
    assert seen_a == [int(i) for order in orders for i in order]
    for epoch in range(epochs):
        chunk = seen_a[epoch * n:(epoch + 1) * n]
        assert sorted(chunk) == list(range(n))


def test_default_flags_match_full_permutation(capsys):
    """--shuffle-block 0 (the default) is the old rng.permutation order."""
    from ml import train_bc_policy as T
    args = T.parse_args(["--data", "x"])
    assert args.shuffle_block == 0
    assert args.cache_dtype == "float32"
    n, epochs, seed, batch = 23, 2, 0, 8
    expected = _old_orders(n, epochs, seed)
    seen = []
    T.train_model(_seen_items(n, seen), epochs, 1e-3, batch, seed,
                  shuffle_block=args.shuffle_block)
    got = [seen[epoch * n:(epoch + 1) * n] for epoch in range(epochs)]
    assert all(np.array_equal(np.asarray(g), e) for g, e in zip(got, expected))
    # Not passing the flag uses the same default inside train_model.
    seen_default = []
    T.train_model(_seen_items(n, seen_default), epochs, 1e-3, batch, seed)
    assert seen_default == seen
    err = capsys.readouterr().err
    assert "epoch 1/2" in err and "train_loss" in err
    assert "elapsed" in err and "rows/sec" in err

    def on_epoch(epoch, model, history):
        return {"val_loss": 1.25, "val_top1": 0.5, "val_top3": 0.75}

    T.train_model(_seen_items(4, []), 1, 1e-3, 2, seed, on_epoch=on_epoch)
    err = capsys.readouterr().err
    assert "epoch 1/1" in err
    assert "val_loss 1.25" in err and "val_top1 0.5" in err and "val_top3 0.75" in err


def test_help_documents_shuffle_block_and_cache_dtype(capsys):
    from ml import train_bc_policy as T
    with pytest.raises(SystemExit) as exc:
        T.parse_args(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--shuffle-block" in text and "full permutation" in text
    assert "--cache-dtype" in text and "float16" in text and "float32" in text
    assert "--shuffle-block" in T.__doc__ and "--cache-dtype" in T.__doc__
    assert "rows/sec" in T.__doc__


def _option_onehot_slices():
    n_types = len(enc.OPTION_TYPES)
    card = enc._CARD_DIM
    zones = len(enc.ZONES)
    src = n_types + card
    tgt = src + zones + 1
    hp = enc.OPTION_DIM - enc._V4_OPTION - enc._V5_OPTION
    hero = enc.OPTION_DIM - enc._V5_OPTION
    return {
        "type": (0, n_types),
        "src_zone": (src, src + zones),
        "tgt_zone": (tgt, tgt + zones),
        "hp": (hp, hp + enc.HP_VOCAB_DIM),
        "hero": (hero, hero + enc.HERO_VOCAB_DIM),
    }


def test_float16_cache_onehots_exact_and_model_trains(tmp_path):
    from ml import train_bc_policy as T
    from ml.bc_policy import pad_batch
    games = _rows(n_games=4, per_game=2)
    for rows in games:
        for row in rows:
            row["state"]["hero"] = "BG36_HERO_002"
            row["state"]["hero_power"]["card_id"] = "BG36_HERO_000p"
    pattern = _write(tmp_path, games)
    cache = tmp_path / "cache"
    cache.mkdir()
    files = T.data_files([pattern])
    train, held, _info = T.encode_stream(files, set(), BUILD, str(cache),
                                         cache_dtype="float16")
    try:
        assert train.path.endswith("train_options.f16")
        assert train.state_path.endswith("train_states.f16")
        assert held.path.endswith("heldout_options.f16")
        assert held.state_path.endswith("heldout_states.f16")
        assert train.dtype is np.float16 and held.dtype is np.float16
        assert not list(cache.glob("*.f32"))
        assert train.O.dtype == np.float16 and train.S.dtype == np.float16
        nbytes = os.path.getsize(train.path)
        assert nbytes == int(np.prod(train.O.shape)) * 2
        with pytest.raises(ValueError):
            np.memmap(train.path, dtype=np.float32, mode="r", shape=train.O.shape)

        ref = []
        for path in files:
            part = T.encode_file(path, set(), BUILD)["train"]
            off = 0
            for i, n_opt in enumerate(part["n_opt"]):
                ref.append((part["S"][i], part["O"][off:off + n_opt]))
                off += n_opt
        assert len(ref) == len(train) > 0
        slices = _option_onehot_slices()
        hp0 = enc.STATE_DIM - enc.HERO_VOCAB_DIM - enc.HP_VOCAB_DIM
        hero0 = enc.STATE_DIM - enc.HERO_VOCAB_DIM
        saw_one = False
        differed = False
        for i, (state, options) in enumerate(ref):
            got_s, got_o, _c = train.encoded(i)
            got_s = np.asarray(got_s)
            got_o = np.asarray(got_o)
            assert got_s.dtype == np.float32 and got_o.dtype == np.float32
            assert np.array_equal(got_s[hp0:hero0], state[hp0:hero0])
            assert np.array_equal(got_s[hero0:], state[hero0:])
            assert got_s[hero0] == np.float32(1.0)
            assert got_s[hp0] == np.float32(1.0)
            saw_one = True
            for lo, hi in slices.values():
                assert np.array_equal(got_o[:, lo:hi], options[:, lo:hi])
            if not np.array_equal(got_o, options):
                differed = True
        assert saw_one and differed

        items = [train.item(i) for i in range(len(train))]
        states, options, *_rest = pad_batch([items[0]])
        assert states.dtype == torch.float32 and options.dtype == torch.float32
        model, history = T.train_model(items, 1, 1e-3, 4, 0)
        assert len(history) == 1 and np.isfinite(history[0])
        assert next(model.parameters()).dtype == torch.float32
    finally:
        train.close()
        held.close()


def test_progress_lines_for_advisor_compare_and_eval(tmp_path, capsys):
    from ml import train_bc_policy as T
    pattern = _write(tmp_path, _rows(n_games=8, per_game=1))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"game_ids": ["g007"]}), encoding="utf-8")
    out = tmp_path / "p.pt"
    rc = T.main(["--data", pattern, "--heldout-ids", str(frozen),
                 "--epochs", "1", "--batch", "4", "--out", str(out)])
    assert rc == 0
    err = capsys.readouterr().err
    epoch_at = err.find("epoch 1/1")
    advisor_at = err.find("advisor-compare: starting")
    eval_at = err.find("eval: starting")
    assert 0 <= epoch_at < advisor_at < eval_at
    assert "train_loss" in err and "elapsed" in err and "rows/sec" in err

    out2 = tmp_path / "p2.pt"
    rc = T.main(["--data", pattern, "--heldout-ids", str(frozen),
                 "--epochs", "1", "--batch", "4", "--out", str(out2),
                 "--no-baseline-evalnet"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "advisor-compare: starting" not in err
    assert "eval: starting" in err and "epoch 1/1" in err
