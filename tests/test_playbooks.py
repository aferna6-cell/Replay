"""Playbook generation tests. A controlled fixture (small StatsDB/kb/emb/
boards) exercises the key-cards/enablers/curve logic precisely; a second
group runs `generate_playbooks()` against the real committed snapshot to
prove it actually produces sensible files for real comps."""

import os

from hsbg_coach import playbooks
from hsbg_coach.cards import CardKnowledge
from hsbg_coach.stats import CompStats, StatsDB


def make_kb():
    return {
        "M1": CardKnowledge(card_id="M1", name="Murloc One", tier=1, attack=1,
                            health=1, tribes=["Murloc"], keywords=[],
                            text="A murloc."),
        "M2": CardKnowledge(card_id="M2", name="Murloc Two", tier=2, attack=2,
                            health=2, tribes=["Murloc"], keywords=[],
                            text="Give your other Murlocs +1/+1."),
        "M3": CardKnowledge(card_id="M3", name="Murloc Enabler", tier=1,
                            attack=1, health=2, tribes=[], keywords=[],
                            text="Whenever you play a Murloc, gain +1/+1."),
    }


def make_stats_db():
    comp = CompStats(name="Murloc Swarm", tribe="Murloc", average_position=3.5,
                     popularity=0.05, core_cards=["Murloc One"], power_turns=[],
                     tier="A")
    return StatsDB(heroes=[], comps=[comp])


BOARDS = [{
    "archetype": "murloc_swarm", "name": "Murloc Swarm", "tribe": "Murloc",
    "boardCount": 50,
    "coreCards": [
        {"name": "Murloc Two", "frequency": 0.9, "commonPosition": 1},
        {"name": "Murloc One", "frequency": 0.5, "commonPosition": 2},
    ],
    "examples": [],
}]

EMB = {"Murloc One": [1.0, 0.0], "Murloc Two": [0.9, 0.1],
      "Murloc Enabler": [0.0, 1.0]}


# --- controlled fixture ---------------------------------------------------

def test_generate_playbooks_writes_one_file_per_comp(tmp_path):
    written = playbooks.generate_playbooks(
        kb=make_kb(), stats_db=make_stats_db(), emb=EMB, boards=BOARDS,
        out_dir=str(tmp_path))
    assert len(written) == 1
    assert os.path.isfile(written[0])
    assert written[0].endswith("murloc-swarm.md")


def test_key_cards_prefer_board_frequency_over_tribe_fallback(tmp_path):
    playbooks.generate_playbooks(kb=make_kb(), stats_db=make_stats_db(),
                                 emb=EMB, boards=BOARDS, out_dir=str(tmp_path))
    text = playbooks.load_playbook("Murloc Swarm", dir=str(tmp_path))
    key_section = playbooks._section(text, "Key cards")
    # Board frequency order (Murloc Two first) must win over the comp's own
    # tribe-strength core_cards list (which would have put Murloc One first).
    assert key_section.index("Murloc Two") < key_section.index("Murloc One")
    assert "real winning-board frequency" in key_section


def test_key_cards_fall_back_to_tribe_strength_without_board_data(tmp_path):
    playbooks.generate_playbooks(kb=make_kb(), stats_db=make_stats_db(),
                                 emb=EMB, boards=[], out_dir=str(tmp_path))
    text = playbooks.load_playbook("Murloc Swarm", dir=str(tmp_path))
    key_section = playbooks._section(text, "Key cards")
    assert "Murloc One" in key_section
    assert "tribe-strength fallback" in key_section


def test_enablers_section_includes_buff_and_cares_cards(tmp_path):
    playbooks.generate_playbooks(kb=make_kb(), stats_db=make_stats_db(),
                                 emb=EMB, boards=BOARDS, out_dir=str(tmp_path))
    text = playbooks.load_playbook("Murloc Swarm", dir=str(tmp_path))
    enablers_section = playbooks._section(text, "Enablers")
    assert "Murloc Enabler" in enablers_section


def test_enablers_for_tribe_excludes_named_cards():
    kb = make_kb()
    out = playbooks.enablers_for_tribe("Murloc", kb, exclude=["Murloc One"])
    assert "Murloc One" not in out
    assert "Murloc Two" in out and "Murloc Enabler" in out
    # cheapest tier first
    assert out.index("Murloc Enabler") < out.index("Murloc Two")


def test_enablers_for_tribe_no_tribe_or_kb_returns_empty():
    assert playbooks.enablers_for_tribe(None, make_kb()) == []
    assert playbooks.enablers_for_tribe("Murloc", {}) == []


def test_enablers_for_tribe_dedupes_cards_sharing_a_display_name():
    # Two different card_ids (reprint/skin variants) can share one display
    # name — the real KB has ~31 of these. A shared name must appear once.
    kb = {
        "M2_BASE": CardKnowledge(card_id="M2_BASE", name="Murloc Two", tier=2,
                                 attack=2, health=2, tribes=["Murloc"],
                                 keywords=[], text="Give your other Murlocs +1/+1."),
        "M2_SKIN": CardKnowledge(card_id="M2_SKIN", name="Murloc Two", tier=2,
                                 attack=2, health=2, tribes=["Murloc"],
                                 keywords=[], text="Give your other Murlocs +1/+1."),
    }
    out = playbooks.enablers_for_tribe("Murloc", kb)
    assert out == ["Murloc Two"]


def test_all_sections_present(tmp_path):
    playbooks.generate_playbooks(kb=make_kb(), stats_db=make_stats_db(),
                                 emb=EMB, boards=BOARDS, out_dir=str(tmp_path))
    text = playbooks.load_playbook("Murloc Swarm", dir=str(tmp_path))
    for header in ("Identity", "Key cards", "Enablers", "Curve",
                  "How to pilot", "Pivot signals"):
        assert f"## {header}" in text
    assert "Murloc Swarm" in playbooks._section(text, "Pivot signals")


def test_load_playbook_missing_returns_none(tmp_path):
    assert playbooks.load_playbook("Nonexistent Comp", dir=str(tmp_path)) is None


def test_playbook_summary_for_prompt_compact_and_bounded(tmp_path):
    playbooks.generate_playbooks(kb=make_kb(), stats_db=make_stats_db(),
                                 emb=EMB, boards=BOARDS, out_dir=str(tmp_path))
    summary = playbooks.playbook_summary_for_prompt(
        ["Murloc Swarm", "Nonexistent Comp"], dir=str(tmp_path), max_chars=3000)
    assert "Murloc Swarm" in summary
    assert len(summary) <= 3000

    tiny = playbooks.playbook_summary_for_prompt(
        ["Murloc Swarm"], dir=str(tmp_path), max_chars=5)
    assert tiny == ""                      # first block alone exceeds budget


def test_load_validated_experiments_none_then_reads_file(tmp_path):
    assert playbooks.load_validated_experiments(dir=str(tmp_path)) is None
    exp_path = os.path.join(str(tmp_path), playbooks.EXPERIMENTS_FILE)
    with open(exp_path, "w", encoding="utf-8") as fh:
        fh.write("# Validated experiments\n- Buying X early on Y comp: PASS\n")
    content = playbooks.load_validated_experiments(dir=str(tmp_path))
    assert "Validated experiments" in content


# --- real committed data ---------------------------------------------------

def test_generate_playbooks_against_real_committed_data(tmp_path):
    written = playbooks.generate_playbooks(out_dir=str(tmp_path))
    assert len(written) >= 20                     # 27 comps in the committed snapshot
    names = {os.path.basename(p) for p in written}
    assert "murloc-scam.md" in names
    assert "undead-forsaken.md" in names

    text = playbooks.load_playbook("Murloc Scam", dir=str(tmp_path))
    assert text is not None
    for header in ("Identity", "Key cards", "Enablers", "Curve",
                  "How to pilot", "Pivot signals"):
        assert f"## {header}" in text
    assert "Bile Spitter" in text                  # a real Murloc Scam core card
