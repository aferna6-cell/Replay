"""The HSReplay guide page renders comps, heroes, trinkets and Aidan's notes."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "render_hsreplay_guides",
    Path(__file__).resolve().parents[1] / "scripts" / "render_hsreplay_guides.py")
render_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(render_mod)


def test_page_data_has_guides_notes_setups_and_heroes():
    data = render_mod.build_data()
    comps = {c["name"]: c for c in data["comps"]}
    assert len(comps) >= 15 and len(data["trinkets"]) >= 100 and len(data["heroes"]) >= 80
    beetles = comps["Beasts - Beetles"]
    assert "Ravaging Scorpid" in beetles["how"]            # HSReplay text, verbatim
    assert beetles["setups"] and beetles["heroes"]
    shop_buff = comps["Demons - Shop Buff"]
    assert shop_buff["notes"] and "Crater Miner" in shop_buff["support_cards"]
    jarax = next(h for h in data["heroes"] if h["name"] == "Lord Jaraxxus")
    assert any(c["name"] == "Demons - Shop Buff" for c in jarax["comps"])


def test_page_renders_without_raw_placeholders():
    html = render_mod.render(render_mod.build_data())
    assert "__DATA__" not in html and "HSBG Comp Guides" in html
    assert 'id="heroes"' in html and "Aidan" in html
