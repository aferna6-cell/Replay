"""ml/lora_dataset.py — corpus -> SFT dataset. Pure stdlib, no torch."""

import json
import os

from ml.lora_dataset import build_sft_dataset


def _write_corpus(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _row(turn, verdict, better="", move="Buy Monstrous Macaw", chosen_source="director"):
    return {
        "state": {"turn": turn, "tavern_tier": 3, "gold": 6, "hero_health": 30,
                  "hero": "HERO_01", "board": [{"name": "Alley Cat", "attack": 1, "health": 1}]},
        "candidates": [{"action": "buy", "detail": {"card": "Monstrous Macaw"}},
                       {"action": "roll", "detail": {}}],
        "chosen": {"move": move, "why": "synergy", "source": chosen_source},
        "verdict": verdict,
        "placement": 2,
        "better": better,
    }


def test_missing_corpus_yields_empty_dataset(tmp_path):
    counts = build_sft_dataset(str(tmp_path / "no-such-corpus.jsonl"),
                               str(tmp_path / "out.jsonl"))
    assert counts["written"] == 0
    assert counts["total_lines"] == 0
    assert os.path.isfile(tmp_path / "out.jsonl")   # still writes an (empty) file


def test_good_verdict_included_as_director_source(tmp_path):
    corpus = str(tmp_path / "corpus.jsonl")
    out = str(tmp_path / "out.jsonl")
    _write_corpus(corpus, [_row(5, "good")])
    counts = build_sft_dataset(corpus, out)
    assert counts["written"] == 1
    assert counts["included_good"] == 1
    assert counts["included_rewritten"] == 0
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "director"
    assert len(row["messages"]) == 3
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["role"] == "user"
    assistant = json.loads(row["messages"][2]["content"])
    assert assistant["move"] == "Buy Monstrous Macaw"


def test_bad_verdict_without_better_is_skipped(tmp_path):
    corpus = str(tmp_path / "corpus.jsonl")
    out = str(tmp_path / "out.jsonl")
    _write_corpus(corpus, [_row(5, "bad", better="")])
    counts = build_sft_dataset(corpus, out)
    assert counts["written"] == 0
    assert counts["skipped"] == 1


def test_bad_verdict_with_better_is_rewritten(tmp_path):
    corpus = str(tmp_path / "corpus.jsonl")
    out = str(tmp_path / "out.jsonl")
    _write_corpus(corpus, [_row(5, "bad", better="Sell Deflect-o-Bot")])
    counts = build_sft_dataset(corpus, out)
    assert counts["written"] == 1
    assert counts["included_rewritten"] == 1
    row = json.loads(open(out).readline())
    assert row["source"] == "hindsight_rewrite"
    assistant = json.loads(row["messages"][2]["content"])
    assert assistant["move"] == "Sell Deflect-o-Bot"
    assert "hindsight" in assistant["why"].lower()


def test_questionable_and_unknown_excluded_by_default(tmp_path):
    corpus = str(tmp_path / "corpus.jsonl")
    out = str(tmp_path / "out.jsonl")
    _write_corpus(corpus, [_row(1, "questionable"), _row(2, "unknown")])
    counts = build_sft_dataset(corpus, out)
    assert counts["written"] == 0
    assert counts["skipped"] == 2


def test_min_verdict_questionable_includes_questionable_rows(tmp_path):
    corpus = str(tmp_path / "corpus.jsonl")
    out = str(tmp_path / "out.jsonl")
    _write_corpus(corpus, [_row(1, "questionable"), _row(2, "unknown")])
    counts = build_sft_dataset(corpus, out, min_verdict="questionable")
    assert counts["written"] == 1     # only the questionable one; unknown never qualifies
    assert counts["skipped"] == 1


def test_dedup_identical_states(tmp_path):
    corpus = str(tmp_path / "corpus.jsonl")
    out = str(tmp_path / "out.jsonl")
    same_state_rows = [_row(5, "good"), _row(5, "good")]   # identical "state" dicts
    _write_corpus(corpus, same_state_rows)
    counts = build_sft_dataset(corpus, out)
    assert counts["written"] == 1
    assert counts["deduped"] == 1


def test_malformed_line_is_skipped_not_raised(tmp_path):
    corpus = str(tmp_path / "corpus.jsonl")
    out = str(tmp_path / "out.jsonl")
    os.makedirs(tmp_path, exist_ok=True)
    with open(corpus, "w", encoding="utf-8") as fh:
        fh.write("not valid json\n")
        fh.write(json.dumps(_row(9, "good")) + "\n")
    counts = build_sft_dataset(corpus, out)
    assert counts["total_lines"] == 2
    assert counts["parsed"] == 1
    assert counts["written"] == 1


def test_cli_help_does_not_error(capsys):
    from ml.lora_dataset import _cli
    try:
        _cli(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
