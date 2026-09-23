#!/usr/bin/env python3
"""Render HSReplay comp, hero and trinket guides as one browsable HTML page.

Guide text is verbatim from HSReplay as pulled by
``scripts/ingest_hsreplay_guides.py`` (comps / heroes / trinkets .json): tier,
difficulty, summary, how to play, when to commit, common enablers, core and
add-on cards, hero + buddy guides, trinket guides and stats. The only markup is
turning HSReplay's ``[[Card||dbf]]`` tags into card chips.

Clearly labelled extras sit beside it, never mixed into HSReplay's text:
  * Aidan's notes + support cards (data/hsreplay_guides/aidan_notes.json)
  * trinket setups: trinkets whose HSReplay guide names the comp's cards/tribe
  * heroes whose HSReplay guide names the comp's cards or tribe

Refresh straight from HSReplay, then render (PowerShell or bash):

    python scripts/ingest_hsreplay_guides.py --workers 12
    python scripts/render_hsreplay_guides.py          # -> data/hsreplay_guides/guides.html

Open the HTML file in any browser.
"""
import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GUIDES = ROOT / "data" / "hsreplay_guides"
POOL = ROOT / "data" / "cards" / "bg_live_pool_36_6_1.json"
MARK = re.compile(r"\[\[([^\]|]+)(?:\|\|\d+)?\]\]")
TIER = {1: "S", 2: "A", 3: "B"}


def _live_names() -> set:
    if not POOL.is_file():
        return set()
    doc = json.loads(POOL.read_text(encoding="utf-8"))
    return set(doc.get("names") or [m["name"] for m in doc.get("minions", [])])


def build_data() -> dict:
    live = _live_names()

    def rich(text: str) -> str:
        out, last = [], 0
        text = text or ""
        for m in MARK.finditer(text):
            out.append(html.escape(text[last:m.start()]))
            name = m.group(1).strip()
            cls = "card" if (not live or name in live) else "card oop"
            out.append(f'<span class="{cls}">{html.escape(name)}</span>')
            last = m.end()
        out.append(html.escape(text[last:]))
        return "".join(out).replace("\n", "<br>")

    def cards(items) -> list:
        return [{"name": x.get("name"), "live": x.get("in_live_pool")}
                for x in items or [] if isinstance(x, dict) and x.get("name")]

    from hsbg_coach.comp_notes import load_notes, notes_for, support_cards
    from hsbg_coach.hero_comps import all_fits
    from hsbg_coach.lobby_playbook import live_comp_infos

    comps_doc = json.loads((GUIDES / "comps.json").read_text(encoding="utf-8"))
    trinkets_doc = json.loads((GUIDES / "trinkets.json").read_text(encoding="utf-8"))
    infos = {ci.name: ci for ci in live_comp_infos()}
    fits = all_fits()
    rules = load_notes().get("support_rules") or {}

    # Trinket setups: HSReplay trinket guides that name the comp's cards/tribe.
    tguides = []
    for t in trinkets_doc.get("trinkets") or []:
        g = t.get("guide_text") or ""
        if g:
            tguides.append((t, set(MARK.findall(g)), MARK.sub(lambda m: m.group(1), g).lower()))

    def setups(ci) -> list:
        out = []
        word = ci.tribe.lower()
        for t, named, low in tguides:
            hit = [n for n in ci.cards if n in named]
            if hit or re.search(rf"\b{word}s?\b", low):
                s_ = t.get("stats") or {}
                out.append({"name": t.get("name"), "type": t.get("type"),
                            "tier": (s_.get("tier") or "").upper() or None,
                            "avg": s_.get("avg_final_placement"),
                            "why": ", ".join(hit[:2]) or ci.tribe})
        out.sort(key=lambda x: (x["avg"] is None, x["avg"] or 9))
        return out[:8]

    def heroes_for(ci) -> list:
        out = []
        for h, f in fits:
            if ci.name in f.comps:
                out.append({"name": h["name"], "why": "names " + ", ".join(f.comps[ci.name][:2])})
            elif ci.tribe in f.tribes:
                out.append({"name": h["name"], "why": f"favors {ci.tribe}"})
        return out

    comps = []
    for c in comps_doc.get("comps") or []:
        tribe = (c.get("tribe") or "").strip()
        ci = infos.get(c.get("name"))
        note_rules = ((load_notes().get("comps") or {}).get(c.get("name")) or {}).get("also_buy") or []
        comps.append({
            "notes": [html.escape(n) for n in notes_for(c.get("name"))],
            "support": [{"label": (rules.get(r) or {}).get("label", r)} for r in note_rules],
            "support_cards": support_cards(c.get("name")) if note_rules else [],
            "setups": setups(ci) if ci else [],
            "heroes": heroes_for(ci) if ci else [],
            "name": c.get("name"), "tribe": tribe,
            "tier": TIER.get(int(c.get("tier") or 3), "B"),
            "difficulty": c.get("difficulty"),
            "summary": html.escape(c.get("summary") or ""),
            "how": rich(c.get("how_to_play")), "commit": rich(c.get("when_to_commit")),
            "enablers_text": rich(c.get("common_enablers_text")),
            "core": cards(c.get("core_cards")), "addons": cards(c.get("addon_cards")),
            "url": c.get("source_url"), "updated": (c.get("last_updated") or "")[:10],
        })
    trinkets = []
    for t in trinkets_doc.get("trinkets") or []:
        s = t.get("stats") or {}
        trinkets.append({
            "name": t.get("name"), "type": t.get("type"),
            "tier": (s.get("tier") or "").upper() or None,
            "avg": s.get("avg_final_placement"), "pick": s.get("pick_rate"),
            "top1": s.get("top1_avg_final_placement"),
            "effect": html.escape(t.get("effect_summary") or "").replace("\n", " "),
            "guide": rich(t.get("guide_text") or ""),
            "recent": bool(t.get("guide_recently_updated")),
        })
    heroes_doc = json.loads((GUIDES / "heroes.json").read_text(encoding="utf-8"))
    fit_by = {h.get("name"): f for h, f in fits}
    heroes = []
    for h in heroes_doc.get("heroes") or []:
        st = h.get("stats") or {}
        f = fit_by.get(h.get("name"))
        heroes.append({
            "name": h.get("name"), "url": h.get("source_url"),
            "tier": (st.get("tier_v2") or "").upper() or None,
            "avg": st.get("avg_final_placement"), "pick": st.get("pick_rate"),
            "guide": rich(h.get("guide_text") or ""),
            "buddy": rich(h.get("buddy_guide_text") or ""),
            "comps": [{"name": k, "cards": v} for k, v in (f.comps.items() if f else [])],
            "tribes": list(f.tribes) if f else [], "avoid": list(f.avoid) if f else [],
            "buys": list(f.buys) if f else [],
        })
    return {
        "heroes": heroes,
        "as_of": comps_doc.get("as_of"), "patch": comps_doc.get("patch"),
        "listed": comps_doc.get("count_listed"),
        "dropped": len(comps_doc.get("dropped") or []),
        "comps": comps, "trinkets": trinkets,
    }


def render(data: dict, full_document: bool = True) -> str:
    body = TEMPLATE_PATH.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    if not full_document:
        return body
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "</head><body>" + body + "</body></html>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default=str(GUIDES / "guides.html"))
    ap.add_argument("--fragment", action="store_true",
                    help="omit <html>/<head>/<body> (for embedding)")
    args = ap.parse_args(argv)
    data = build_data()
    Path(args.out).write_text(render(data, not args.fragment), encoding="utf-8")
    print(f"{len(data['comps'])} comps, {len(data['trinkets'])} trinkets -> {args.out}")
    return 0


TEMPLATE_PATH = ROOT / "scripts" / "templates" / "hsreplay_guides.html"


if __name__ == "__main__":
    raise SystemExit(main())
