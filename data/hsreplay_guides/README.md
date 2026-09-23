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
