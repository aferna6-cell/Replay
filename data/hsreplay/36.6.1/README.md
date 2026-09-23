# HSReplay local DB snapshot (v0.1)

Patch: 36.6.1 (Real 36.6.1 / Season 14)
Built: see meta.json scraped_at

## Files
- meta.json — patch, counts, train weights, gaps
- comps.json — highest train weight (guides + cores/addons/enablers)
- heroes.json — guides + pick_rate (placement stats need Tier7 auth)
- trinkets.json — guides + pick/placement stats
- minions.json — live lobby pool only (naga excluded)
- spells.json — tavern-tier BG spells (no HSReplay guides; manual OK)
- tribes.json — lobby tribes + top comps

## Train weights
1. comps = highest
2. heroes + trinkets = support (affinity to comps learned in training)
3. old patch = 0

## Refresh
Re-run build_snapshot / scrape when HSReplay updates; keep old patch dirs at weight 0.
