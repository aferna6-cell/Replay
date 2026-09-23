"""Build data/cards/bg_card_names.json — every Battlegrounds card id → name,
type, tier and tribes, from HearthstoneJSON.

The live coach's card KB (bg_cards.json) only holds the current minion pool,
so shop cards outside it (tavern spells, dual-tribe cards dropped with Naga,
cards HearthstoneJSON doesn't flag as pool minions) showed up as raw ids like
"BG21_018". This map is the fallback for names and tribes.

  python scripts/build_bg_card_names.py                  # fetch latest
  python scripts/build_bg_card_names.py --cards cards.json   # offline file
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cards" / "bg_card_names.json"
URL = "https://api.hearthstonejson.com/v1/latest/enUS/cards.json"
_RACE = {"MECHANICAL": "Mech", "ELEMENTAL": "Elemental", "BEAST": "Beast",
         "DEMON": "Demon", "DRAGON": "Dragon", "MURLOC": "Murloc",
         "PIRATE": "Pirate", "QUILBOAR": "Quilboar", "UNDEAD": "Undead",
         "NAGA": "Naga", "ALL": "All"}
_TYPES = {"MINION", "BATTLEGROUND_SPELL", "BATTLEGROUND_TRINKET",
          "BATTLEGROUND_ANOMALY", "BATTLEGROUND_QUEST_REWARD", "HERO"}


def is_bg(c: dict) -> bool:
    cid = c.get("id") or ""
    if not c.get("name"):
        return False
    if c.get("type") == "SPELL" and (c.get("set") == "BATTLEGROUNDS"
                                     or cid.startswith(("BG", "TB_Bacon"))):
        return True                  # Bananas & co. (BG_EX1_014t)
    if c.get("type") not in _TYPES:
        return False
    return (cid.startswith(("BG", "TB_Bacon")) or c.get("techLevel") is not None
            or c.get("isBattlegroundsPoolMinion") or c.get("battlegroundsHero"))


def build(cards: list) -> dict:
    out = {}
    for c in cards:
        if not is_bg(c):
            continue
        races = c.get("races") or ([c["race"]] if c.get("race") else [])
        out[c["id"]] = {
            "name": c["name"], "type": c["type"],
            "tier": c.get("techLevel"),
            "tribes": [_RACE.get(r, r.title()) for r in races],
        }
    return dict(sorted(out.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cards", help="local HearthstoneJSON cards.json")
    ap.add_argument("-o", "--out", default=str(OUT))
    args = ap.parse_args(argv)
    if args.cards:
        cards = json.loads(Path(args.cards).read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(URL, timeout=60) as r:
            cards = json.loads(r.read().decode("utf-8"))
    names = build(cards)
    Path(args.out).write_text(json.dumps(
        {"_source": URL, "_count": len(names), "cards": names},
        ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{len(names)} Battlegrounds cards -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
