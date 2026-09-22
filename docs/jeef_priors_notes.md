# Jeef VOD soft priors (Dark Gift / Activate / HP / Spells)

Distilled from `/workspace/hsreplay-tier7/jeef_vod_pilot/labels` (+ RDU rows in
the same corpus). **Sample is thin** — these are soft placement nudges in
`hsbg_coach/jeef_priors.py`, not hard rules or training changes.

## Dark Gift (Power.log calibrated 2026-09)

| Signal | Evidence | Soft prior / detection |
|---|---|---|
| Button id | `cardId=BG36_Button_DarkGift`, `entityName=Dark Discovery` | Match calibrated id/name (not `TB_BaconShop_DarkGift_Button`) |
| Legal | DebugPrintOptions `error=NONE` | Emit when usable |
| Blocked | `REQ_ENOUGH_MANA`, `REQ_NOT_EXHAUSTED_ACTIVATE` | Gate via gold + `EXHAUSTED` / `BACON_DARK_GIFT_PRESSABLE_VFX` |
| Related tags | `HAS_DARK_GIFT`, `DARK_GIFT_ENTITY`, `BG36_MidGameEffect_*` | Chrome / gifted minions — not the button |
| Prefer gift over early level | Jeef ASR: "Should have just dark gifted…" | Nudge when gold covers gift but thin level empties the turn |
| Midgame window | Discover @ turn≈4, tier≈2, gold≈3 | Extra boost turn 3–7, tier ≤3 |
| Gift + HP same shop | Labeled plan `dark_gift_chef_hp` | Extra boost when HP also usable |
| Panic / late | (inferred) | Demote HP≤12 thin board, or turn≥12 full board |

## Activate / Hero power (Power.log calibrated 2026-09)

| Signal | Evidence | Soft prior / detection |
|---|---|---|
| Activate tag | `HAS_ACTIVATE_POWER=1` on board minion | Required |
| Cost | `TAG_SCRIPT_DATA_NUM_1` when present (else `COST`) | e.g. Suspicious Prisonguard cost 3 |
| Filter | Friendly `zone=PLAY`, `zonePos>=1`, real `BG*`/`BGS_*` minion | **Exclude** `TB_BaconShop_*` Refresh/Freeze/DragBuy/Tier buttons (they also set `HAS_ACTIVATE_POWER`) |
| `BACON_TRIGGER_XY` | Rare on real activates (≪1%) | **Do not require** (was wrong as Activate gate) |
| `BACON_TRIGGER_UPBEAT` | Present on trinkets/magic items (e.g. Trip Vouchers) | Not an Activate signal |
| 0-cost Activate | ASR/cues | Strong boost for cost 0 |
| Clickable HP | Labeled HP decisions | Baseline boost when `hero_power.usable` |

## Spells (BUY_SPELL)

| Spell | Evidence | Soft prior |
|---|---|---|
| Corrupted Coin `BG36_303` | Repeated Aberration VOD frames/ASR | Known table + Jeef bonus |
| Shop `BATTLEGROUND_SPELL` ids | Power.log 2026-09 slices (Careful Investment, Fortify, Queen's Command, …) | Mild `spell_roles` entries |
| Strike Oil / Warband Whistle | Sparse OCR | Name-substring hints only |
| Unknown spells | — | Remain demoted via spell_roles (`_GENERIC_BONUS`) |

## Counts (approx.)

- Dark-gift-related decisions: **3**
- Hero-power decisions: **2**
- Activate ASR hits: sparse
- Explicit BUY_SPELL cardId labels: **≈0** (shop table extended from live Power.log)
