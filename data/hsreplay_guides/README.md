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
```

Patch target: **36.6.1** Aberrations. Live pool gate: `data/cards/bg_live_pool_36_6_1.json` (252 minions).

## Live playbook (hard NEXT gates)
`hsbg_coach/lobby_playbook.py` turns these files into Aidan's locked sequence.
It is **internal strategy**: it only changes NEXT. The overlay shows NEXT plus
short alternates, with no LOBBY / ENABLERS / PLAN header lines
(`overlay_lines()` still exists for debugging).

1. **HERO** — hero-guide HP / buy-pref actions can lead NEXT
2. **LOBBY** — lobby tribes ranked by best HSReplay comp tier
3. **ENABLERS** — `when_to_commit` + enabler cards of the strong tribes' comps
   (live pool only)
4. **FILL** — until committed: Roll can't be NEXT on sparse + solid shop; a
   garbage buy can't lead while a solid fill is up; an enabler in shop is NEXT
5. **COMMIT** — first clear enabler hit (shop / hand / board / discover) locks
   the comp for the rest of the game. After the lock, NEXT hunts only that
   comp's HSReplay core / key / trigger cards: a core buy leads (over roll,
   level, fills), an unaffordable one is frozen, off-plan buys (other tribes,
   neutral flex, other comps' enablers) never lead or appear as the first
   alternate, and PLAN pieces are never sold.
