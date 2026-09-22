# Jeef VOD soft priors (Dark Gift / Activate / HP / Spells)

Distilled from `/workspace/hsreplay-tier7/jeef_vod_pilot/labels` (+ RDU rows in
the same corpus). **Sample is thin** — these are soft placement nudges in
`hsbg_coach/jeef_priors.py`, not hard rules or training changes.

## Dark Gift

| Signal | Evidence | Soft prior |
|---|---|---|
| Prefer gift over early level | Jeef ASR: "Should have just dark gifted. Level is kind of dumb." | Nudge gift when gold covers gift but a thin level would empty the turn |
| Midgame window | Discover @ turn≈4, tier≈2, gold≈3 (Replication / Treasure Parrot) | Extra boost turn 3–7, tier ≤3 |
| Gift + HP same shop | Labeled plan `dark_gift_chef_hp` | Extra boost when HP also usable and affordable |
| Panic / late | (inferred) | Demote when HP≤12 with thin board, or turn≥12 full board |

**Gap:** No `ActionKind=dark_gift` on button press — gift appears as ASR, discover
`dark_gift` midgame-effect ids, or `comp_plan`. Button cardId may need live
Power.log `# CALIBRATE` (we match name/`DarkGift`/`gift`+`button` patterns).

## Activate / Hero power

| Signal | Evidence | Soft prior |
|---|---|---|
| 0-cost Activate | ASR/cues ("activate zero…") | Strong boost for cost 0 |
| Clickable HP | 2 labeled HP decisions; ASR values HP with gift/chef | Baseline boost when `hero_power.usable`; extra with gift |

**Gap:** Almost no gold/tier/board on HP/activate decision rows.

## Spells (BUY_SPELL)

| Spell | Evidence | Soft prior |
|---|---|---|
| Corrupted Coin `BG36_303` | Repeated Aberration VOD frames/ASR | Known table + Jeef bonus |
| Strike Oil / Warband Whistle | Sparse OCR | Name-substring hints only |
| Unknown spells | — | Remain demoted via `spell_roles` (`_GENERIC_BONUS`) |

**Gap:** ≈0 explicit BUY_SPELL decisions with cardIds in the Jeef JSONL. Frequency
table is essentially Corrupted Coin from frames, not a rich buy histogram.

## Counts (approx.)

- Dark-gift-related decisions: **3**
- Hero-power decisions: **2**
- Activate ASR hits: sparse
- Explicit BUY_SPELL cardId labels: **≈0**
