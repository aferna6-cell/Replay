"""Overlay tests — exercise the pure text formatter (no display needed)."""

from hsbg_coach.overlay import format_overlay_text


def _snap(**over):
    base = {
        "turn": 5, "phase": "recruit", "tavern_tier": 2, "gold": 8,
        "hero_health": 27,
        "board": [{"position": 1, "name": "Tabbycat", "attack": 1, "health": 1}],
        "shop": [{"name": "Rockpool Hunter", "attack": 2, "health": 3}],
        "notes": [],
    }
    base.update(over)
    return base


def test_renders_header_and_board():
    text = format_overlay_text(_snap())
    assert "turn 5" in text
    assert "tier 2" in text
    assert "gold 8" in text
    assert "Tabbycat 1/1" in text
    assert "Rockpool Hunter 2/3" in text


def test_includes_odds_when_provided():
    text = format_overlay_text(_snap(), odds="win 62% / tie 8% / loss 30%")
    assert "Combat: win 62%" in text


def test_empty_board_shows_placeholder():
    text = format_overlay_text(_snap(board=[]))
    assert "(empty)" in text


def test_notes_are_shown():
    text = format_overlay_text(_snap(notes=["local_player not yet identified"]))
    assert "local_player not yet identified" in text


def test_handles_missing_fields_gracefully():
    # A sparse snapshot (early game, nothing known yet) must not crash.
    text = format_overlay_text({"phase": "hero_select"})
    assert "hero_select" in text
