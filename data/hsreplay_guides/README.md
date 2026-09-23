# HSReplay Battlegrounds guides (ingested)

Committed snapshots of **HSReplay** How-to-Play / Hero Guide / Comp Guide /
Trinket Guide text for the live coach. Strategy copy is cited from HSReplay
only — the coach never invents guide text.

## Files
- `heroes.json` (+ `heroes/{dbf_id}_{slug}.json`) — hero guide + buddy guide +
  structured `hp` / `cycle` / `buy_prefs` bullets
- `comps.json` — live-pool comps (Naga / out-of-pool dropped) with
  `how_to_play`, `when_to_commit`, `key_names`, core/enabler cards
- `trinkets.json` — every trinket with `guide_text` (when present) + stats
- `manifest.json` — ingest stamp

## Refresh
```bash
export HSREPLAY_COOKIE_FILE=/path/to/hsreplay.cookies   # Netscape or header
python scripts/ingest_hsreplay_guides.py --workers 12
python scripts/render_hsreplay_guides.py   # browsable page -> guides.html
```
`guides.html` shows the comp and trinket guides exactly as HSReplay publishes
them (not committed; regenerate after each ingest).

Patch target: **36.6.1** Aberrations. Live pool gate: `data/cards/bg_live_pool_36_6_1.json` (252 minions).

## Live playbook (hard NEXT gates)
`hsbg_coach/lobby_playbook.py` turns these files into Aidan's locked sequence.
It is **internal strategy**: it only changes NEXT. The overlay shows NEXT plus
short alternates, with no LOBBY / ENABLERS / PLAN header lines
(`overlay_lines()` still exists for debugging).

1. **HERO** — hero-guide HP / buy-pref actions can lead NEXT
2. **LOBBY** — lobby tribes ranked by best HSReplay comp tier
3. **ENABLERS** — enablers and core are one set per comp: `when_to_commit` +
   enabler cards + key/core cards (live pool only). Only S/A comps of the
   strong lobby tribes are preloaded; S beats A; a B comp locks only on a high
   roll (3+ of its cards already owned)
4. **FILL** — until committed: Roll can't be NEXT on sparse + solid shop; a
   garbage buy can't lead while a solid fill is up; an enabler in shop is NEXT
5. **COMMIT** — first clear enabler/core hit (shop / hand / board / discover) locks
   the comp for the rest of the game. After the lock, NEXT hunts only that
   comp's HSReplay core / key / trigger cards: a core buy leads (over roll,
   level, fills), an unaffordable one is frozen, off-plan buys (other tribes,
   neutral flex, other comps' enablers) never lead or appear as the first
   alternate, and PLAN pieces are never sold.

## Aidan's notes + heroes
- `aidan_notes.json` — your own additions per comp (free-text `notes`, plus
  `also_buy` rules that match card text, e.g. spell / Blood Gem generators for
  Shop Buff Demons, Discover cards for APM Pirates). After a comp locks, matching
  cards count as on-plan and are bought over rolling when no core card is up.
  Shown on the guide page in a dashed "Aidan's notes" box, separate from HSReplay.
- Heroes (`hsbg_coach/hero_comps.py`) — read from HSReplay hero/buddy guides:
  comps whose enabler/core cards the guide names, tribes it favors or says to
  avoid, and buy-preference cards. The coach uses this to choose which lobby
  tribes to hunt and to break ties between comps of the same HSReplay tier;
  hero guide buys also stay on-plan after the lock.

## Hero + trinket picks: most 1st places
- Hero select (`hsbg_coach/draft.py::rank_heroes`) ranks by HSReplay's
  1st-place rate (`final_placement_distribution[0]`), highest first; average
  placement only breaks ties. The weakest 1st rate is the reroll. A lobby-fit
  nudge of a couple of points applies when lobby tribes are known. A hero
  whose HSReplay guide calls its hero power weak loses 6 points
  (`hsbg_coach/hero_power_verdict.py`) and is named in the reroll line.
- Trinkets (`rank_trinkets`) rank by 1st-place rate when HSReplay's trinket
  placement distribution is ingested, otherwise by average placement (shown as
  `avg 3.88`), adjusted by board / guide fit. The ingest keeps
  `final_placement_distribution` for trinkets whenever HSReplay's API returns
  it — re-run the ingest on your PC.
- Internally, average placement converts to the 1st-% scale with a
  least-squares fit over the ingested heroes (`hsbg_coach/first_place.py`),
  so trinkets with and without a distribution compare on one scale.

## Trinkets follow their HSReplay guides (`hsbg_coach/trinket_comps.py`)
HSReplay's trinket `favorable_tribes` is empty for every trinket, so the guide
text is read the same way as hero guides: comps it names in words ("Commit
Attack Scaling Undead", "Leviathan Beasts"), comps whose core cards it names,
tribes it favors or avoids (incl. "Eles" / "Quils"), cards it tells you to buy,
and out-of-patch tribes (Naga-only guides are dead).
- **Pick:** with PLAN locked, a trinket whose guide is for that comp is
  promoted (tribe match less so) and one whose guide wants another comp is
  demoted; before the lock, a guide favoring a strong lobby tribe is promoted
  and one whose tribes are missing from the lobby is demoted.
- **After equip:** its guide steers strong tribes and breaks ties between
  same-tier comps (after the hero guide), and — when the guide agrees with the
  locked comp — the cards it names count as on-plan support buys.
- The guide page's Trinkets tab shows what each guide points to, and each
  comp's "Trinket setups" use the same reading.
