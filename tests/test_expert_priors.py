"""Expert population prior helpers (Firestone top-MMR)."""

import json
from hsbg_coach.firestone_stats import DEFAULT_REFRESH_MMR, EXPERT_MMR_PREFERRED
from hsbg_coach.stats import expert_prior_note, loaded_stats_mmr


def test_expert_mmr_constants():
    assert DEFAULT_REFRESH_MMR in EXPERT_MMR_PREFERRED
    assert 1 in EXPERT_MMR_PREFERRED
    assert 10 in EXPERT_MMR_PREFERRED


def test_loaded_stats_mmr_and_note(tmp_path):
    path = tmp_path / "heroes.json"
    path.write_text(json.dumps({"_mmr": 10, "heroes": []}), encoding="utf-8")
    assert loaded_stats_mmr(str(path)) == 10
    note = expert_prior_note(str(path))
    assert "mmr=10" in note and "expert" in note

    path.write_text(json.dumps({"_mmr": 100, "heroes": []}), encoding="utf-8")
    note = expert_prior_note(str(path))
    assert "ALL-MMR" in note or "mmr=100" in note
