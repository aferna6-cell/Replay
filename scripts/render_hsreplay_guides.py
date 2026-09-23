#!/usr/bin/env python3
"""Render HSReplay comp + trinket guides as one browsable HTML page.

Everything shown is verbatim from HSReplay as pulled by
``scripts/ingest_hsreplay_guides.py`` (comps.json / trinkets.json): tier,
difficulty, summary, how to play, when to commit, common enablers, core and
add-on cards, trinket guides and stats. Nothing is added or rewritten; the
only markup is turning HSReplay's ``[[Card||dbf]]`` tags into card chips.

Refresh straight from HSReplay, then render (PowerShell or bash):

    python scripts/ingest_hsreplay_guides.py --workers 12
    python scripts/render_hsreplay_guides.py          # -> data/hsreplay_guides/guides.html

Open the HTML file in any browser.
"""
import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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

    comps_doc = json.loads((GUIDES / "comps.json").read_text(encoding="utf-8"))
    trinkets_doc = json.loads((GUIDES / "trinkets.json").read_text(encoding="utf-8"))
    comps = []
    for c in comps_doc.get("comps") or []:
        tribe = (c.get("tribe") or "").strip()
        comps.append({
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
    return {
        "as_of": comps_doc.get("as_of"), "patch": comps_doc.get("patch"),
        "listed": comps_doc.get("count_listed"),
        "dropped": len(comps_doc.get("dropped") or []),
        "comps": comps, "trinkets": trinkets,
    }


def render(data: dict, full_document: bool = True) -> str:
    body = TEMPLATE.replace(
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


TEMPLATE = """<title>HSBG Comp Guides</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --ground:#F2F3F6; --surface:#FFFFFF; --sunk:#E8EAF0; --ink:#1B2030; --muted:#5B6173;
  --line:#D6D9E2; --accent:#3A4DB5; --accent-soft:#E3E7FA;
  --s:#A87A0C; --s-soft:#F6EDD5; --a:#1C7A73; --a-soft:#D9EFEC; --b:#6E7382; --b-soft:#E6E7EB;
  --bad:#A8412F; --chip:#EEF0F6;
  --display:"Bricolage Grotesque",ui-sans-serif,system-ui,sans-serif;
  --body:"IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --ground:#11141B; --surface:#191D27; --sunk:#222735; --ink:#E5E7EE; --muted:#9BA1B3;
    --line:#2C3242; --accent:#93A2FF; --accent-soft:#252C4A;
    --s:#E2B545; --s-soft:#3A3120; --a:#4FC2B8; --a-soft:#1B3533; --b:#A3A8B6; --b-soft:#2A2E38;
    --bad:#E08A78; --chip:#232838;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#11141B; --surface:#191D27; --sunk:#222735; --ink:#E5E7EE; --muted:#9BA1B3;
  --line:#2C3242; --accent:#93A2FF; --accent-soft:#252C4A;
  --s:#E2B545; --s-soft:#3A3120; --a:#4FC2B8; --a-soft:#1B3533; --b:#A3A8B6; --b-soft:#2A2E38;
  --bad:#E08A78; --chip:#232838;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font:15px/1.55 var(--body);margin:0}
.wrap{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:28px 64px}
header.top{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:12px 24px;margin-bottom:20px}
h1{font:800 clamp(28px,4.2vw,42px)/1.05 var(--display);letter-spacing:-.02em;margin:0;text-wrap:balance}
.meta{color:var(--muted);font-size:13px;font-family:var(--mono)}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin-bottom:18px}
.tabs button{appearance:none;border:0;background:none;color:var(--muted);font:600 15px var(--body);padding:10px 14px;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}
.tabs button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--accent)}
.tabs button:focus-visible,.filters button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:18px}
.filters .label{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-right:2px}
.filters button{appearance:none;border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:999px;padding:4px 12px;font:500 13px var(--body);cursor:pointer}
.filters button[aria-pressed="true"]{background:var(--ink);color:var(--ground);border-color:var(--ink)}
.filters .sep{width:1px;height:20px;background:var(--line);margin-inline:6px}
.note{color:var(--muted);font-size:13.5px;max-width:78ch;margin:0 0 18px}
.note b{color:var(--ink)}

.comp{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:14px}
.comp-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 12px;margin-bottom:4px}
.tier{display:inline-grid;place-items:center;width:30px;height:30px;border-radius:7px;font:800 17px var(--display);flex:none;align-self:center}
.tier.S{background:var(--s-soft);color:var(--s)} .tier.A{background:var(--a-soft);color:var(--a)} .tier.B{background:var(--b-soft);color:var(--b)}
.comp h2{font:700 21px/1.2 var(--display);margin:0;letter-spacing:-.01em}
.comp .sub{color:var(--muted);font-size:13px}
.comp .sub a{color:var(--accent)}
.summary{margin:2px 0 14px;color:var(--muted);font-style:italic}
.cols{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:18px 28px}
@media (max-width:820px){.cols{grid-template-columns:1fr}}
h3{font:600 12px var(--body);text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin:0 0 6px}
.block + .block{margin-top:14px}
.guide p{margin:0;max-width:68ch}
.card{font-weight:600;color:var(--ink);background:var(--chip);border-radius:4px;padding:0 4px;white-space:nowrap}
.card.oop{text-decoration:line-through;color:var(--muted)}
.hunt{background:var(--sunk);border-radius:8px;padding:12px 14px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:13px;border:1px solid var(--line);background:var(--surface);border-radius:6px;padding:2px 8px}
.chip.oop{text-decoration:line-through;color:var(--muted)}
.tlist{list-style:none;margin:0;padding:0;display:grid;gap:4px}
.tlist li{display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:baseline;font-size:13.5px}
.tl{font:600 11px var(--mono);padding:1px 5px;border-radius:4px;background:var(--chip);color:var(--muted)}
.tl.S{color:var(--s);background:var(--s-soft)} .tl.A{color:var(--a);background:var(--a-soft)}
.tl.F,.tl.D{color:var(--bad)}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--muted);font-size:12.5px}
.why{color:var(--muted);font-size:12px}
.empty{color:var(--muted);font-size:13px}

.tsearch{flex:1 1 240px;min-width:0;max-width:360px;padding:7px 12px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);font:14px var(--body)}
select{padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);font:14px var(--body)}
.trows{display:grid;gap:8px}
.trow{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 16px;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 16px}
.trow h4{margin:0;font:700 16px var(--display);display:flex;flex-wrap:wrap;gap:8px;align-items:baseline}
.trow .kind{font:500 12px var(--body);color:var(--muted)}
.trow .stats{text-align:right;white-space:nowrap}
.trow .effect{grid-column:1/-1;color:var(--muted);font-size:13.5px}
.trow .tguide{grid-column:1/-1;font-size:14px;border-left:2px solid var(--accent);padding-left:10px;margin-top:2px;max-width:80ch}
.src{font:600 10px var(--body);text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-right:4px}
.fresh{font:600 10px var(--body);text-transform:uppercase;letter-spacing:.08em;color:var(--accent)}
.count{color:var(--muted);font-size:13px;margin-left:auto}
@media (max-width:520px){.trow{grid-template-columns:1fr}.trow .stats{text-align:left}}
</style>

<div class="wrap">
  <header class="top">
    <div>
      <h1>HSBG Comp Guides</h1>
      <div class="meta" id="meta"></div>
    </div>
  </header>

  <nav class="tabs" role="tablist">
    <button role="tab" id="tab-comps" aria-selected="true" aria-controls="comps">Comps</button>
    <button role="tab" id="tab-trinkets" aria-selected="false" aria-controls="trinkets">Trinkets</button>
  </nav>

  <section id="comps" role="tabpanel" aria-labelledby="tab-comps">
    <p class="note" id="compNote"></p>
    <div class="filters" id="compFilters"></div>
    <div id="compList"></div>
  </section>

  <section id="trinkets" role="tabpanel" aria-labelledby="tab-trinkets" hidden>
    <div class="filters">
      <input class="tsearch" id="tq" type="search" placeholder="Search trinket, effect or guide text" aria-label="Search trinkets">
      <select id="ttype" aria-label="Trinket size"><option value="">Greater + Lesser</option><option>Greater</option><option>Lesser</option></select>
      <select id="tguide" aria-label="Guide filter"><option value="1">With HSReplay guide</option><option value="">All trinkets</option></select>
      <select id="tsort" aria-label="Sort"><option value="avg">Best avg placement</option><option value="pick">Most picked</option><option value="name">Name</option></select>
      <span class="count" id="tcount"></span>
    </div>
    <div class="trows" id="tlist"></div>
  </section>
</div>

<script>
const DATA = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const chip = x => `<span class="chip${x.live === false ? " oop" : ""}"${x.live === false ? ' title="Out of the live pool"' : ""}>${esc(x.name)}</span>`;
const fmt = (v, d=2) => v == null ? "–" : Number(v).toFixed(d);

$("#meta").textContent = `Verbatim from hsreplay.net · patch ${DATA.patch} · pulled ${DATA.as_of}`;
$("#compNote").innerHTML = `HSReplay's own comp guides: tier, difficulty, summary, how to play, when to commit, common enablers, core and add-on cards — nothing added. Showing the <b>${DATA.comps.length}</b> comps in the live pool; ${DATA.dropped} of ${DATA.listed} listed by HSReplay are Naga, hidden or out of pool. Crossed-out cards are out of the live pool.`;

// Tabs
const tabs = [["tab-comps","comps"],["tab-trinkets","trinkets"]];
function show(id){
  tabs.forEach(([t,p]) => { const on = p === id; $("#"+t).setAttribute("aria-selected", on); $("#"+p).hidden = !on; });
  try { localStorage.setItem("hsbgTab", id); } catch(e){}
}
tabs.forEach(([t,p]) => $("#"+t).addEventListener("click", () => show(p)));
const hashTab = (location.hash || "").slice(1);
let savedTab = null; try { savedTab = localStorage.getItem("hsbgTab"); } catch(e){}
show(hashTab === "trinkets" || (!hashTab && savedTab === "trinkets") ? "trinkets" : "comps");

// Comps
const state = { tier: "", tribe: "" };
const tribes = [...new Set(DATA.comps.map(c => c.tribe))].sort();
function renderFilters(){
  const f = $("#compFilters");
  const btn = (key, val, label) => `<button data-k="${key}" data-v="${esc(val)}" aria-pressed="${state[key] === val}">${esc(label)}</button>`;
  f.innerHTML = `<span class="label">Tier</span>` + btn("tier","","All") + ["S","A","B"].map(t => btn("tier",t,t)).join("")
    + `<span class="sep"></span><span class="label">Tribe</span>` + btn("tribe","","All") + tribes.map(t => btn("tribe",t,t)).join("");
  f.querySelectorAll("button").forEach(b => b.onclick = () => { state[b.dataset.k] = b.dataset.v; renderFilters(); renderComps(); });
}
function renderComps(){
  const list = DATA.comps.filter(c => (!state.tier || c.tier === state.tier) && (!state.tribe || c.tribe === state.tribe));
  $("#compList").innerHTML = list.map(c => `
    <article class="comp">
      <div class="comp-head">
        <span class="tier ${c.tier}" title="HSReplay tier ${c.tier}">${c.tier}</span>
        <h2>${esc(c.name)}</h2>
        <span class="sub">${esc(c.tribe)} · difficulty ${c.difficulty ?? "–"}/5 · updated ${esc(c.updated)} · <a href="${esc(c.url)}" target="_blank" rel="noopener">HSReplay guide</a></span>
      </div>
      ${c.summary ? `<p class="summary">${c.summary}</p>` : ""}
      <div class="cols">
        <div class="guide">
          <div class="block"><h3>How to play</h3><p>${c.how || '<span class="empty">No guide text.</span>'}</p></div>
          <div class="block"><h3>When to commit</h3><p>${c.commit || '<span class="empty">HSReplay lists none.</span>'}</p></div>
          ${c.enablers_text ? `<div class="block"><h3>Common enablers</h3><p>${c.enablers_text}</p></div>` : ""}
        </div>
        <div>
          <div class="hunt">
            <h3>Core cards</h3>
            <div class="chips">${c.core.map(chip).join("") || '<span class="empty">None listed.</span>'}</div>
            ${c.addons.length ? `<h3 style="margin-top:12px">Add-on cards</h3><div class="chips">${c.addons.map(chip).join("")}</div>` : ""}
          </div>
        </div>
      </div>
    </article>`).join("") || `<p class="empty">No comps match these filters.</p>`;
}
renderFilters(); renderComps();

// Trinkets
function renderTrinkets(){
  const q = $("#tq").value.trim().toLowerCase(), type = $("#ttype").value, g = $("#tguide").value, sort = $("#tsort").value;
  let rows = DATA.trinkets.filter(t => (!type || t.type === type) && (!g || t.guide) &&
    (!q || (t.name + " " + t.effect + " " + t.guide.replace(/<[^>]+>/g,"")).toLowerCase().includes(q)));
  rows.sort((a,b) => sort === "name" ? a.name.localeCompare(b.name)
    : sort === "pick" ? (b.pick ?? -1) - (a.pick ?? -1)
    : (a.avg ?? 99) - (b.avg ?? 99));
  $("#tcount").textContent = `${rows.length} shown`;
  $("#tlist").innerHTML = rows.map(t => `
    <div class="trow">
      <h4>${esc(t.name)} <span class="kind">${esc(t.type)}</span>${t.recent ? ' <span class="fresh">guide updated</span>' : ""}</h4>
      <div class="stats"><span class="tl ${t.tier || ""}">${t.tier || "–"}</span> <span class="num">avg ${fmt(t.avg)} · top 1% ${fmt(t.top1)} · pick ${fmt(t.pick,1)}%</span></div>
      <div class="effect">${t.effect}</div>
      ${t.guide ? `<div class="tguide"><span class="src">HSReplay guide</span> ${t.guide}</div>` : ""}
    </div>`).join("") || `<p class="empty">No trinkets match.</p>`;
}
["#tq","#ttype","#tguide","#tsort"].forEach(s => $(s).addEventListener("input", renderTrinkets));
renderTrinkets();
</script>
"""


if __name__ == "__main__":
    raise SystemExit(main())
