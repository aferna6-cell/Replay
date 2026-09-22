# Live Battlegrounds pool + comps — patch 36.6.1 (Aberrations)

**Fetched:** 2026-09-22 ~7:35 PM ET  
**Goal:** Coach S/A comps and `key_minions` ⊆ live shop pool. Never suggest Naga / rotated cards / fantasy names.

## Sources

| Source | What we used | Notes |
|---|---|---|
| [HSReplay comps](https://hsreplay.net/battlegrounds/comps/) | Public `react_context` JSON embedded in the page (`comps` array with `comp_tier` / `comp_core_cards`) | **No login required** for the tier list. Page still lists 2 Naga comps — we drop them. |
| [HSReplay minions](https://hsreplay.net/battlegrounds/minions/) | SPA shell only (no machine-readable pool in HTML) | Pool taken from HSJSON instead; page noted as cross-check URL. |
| [HearthstoneJSON latest cards](https://api.hearthstonejson.com/v1/latest/enUS/cards.json) | `isBattlegroundsPoolMinion && !battlegroundsNormalDbfId && !isBattlegroundsDuosExclusive && race≠NAGA` | **252** shop minions. HSJSON still flags Naga as pool — **hard-excluded**. |
| [hsbg.cards current pool API](https://hsbg.cards/api/v1/cards?pool=current&cardType=minion) | Cross-check (includes tokens; not used as sole authority) | Confirms Aberration in / Naga rotating. |
| Aidan S-tribe lock | Beast, Elemental, Demon, Undead = S | Matches visible HSReplay S comps. |

## Live pool summary

- **#minions:** 252 (Naga excluded)
- **Tribes:** Aberration 26, Quilboar 24, Murloc 24, Neutral 24, Pirate 24, Elemental 24, Demon 22, Dragon 22, Undead 22, Beast 22, Mech 22, All 4
- **Artifact:** `data/cards/bg_live_pool_36_6_1.json` + regenerated `data/cards/bg_cards.json`
- **Naga:** hard-excluded forever (`naga_excluded_forever: true`)

## Top comps (verified in-pool)

### S (HSReplay tier 1, visible)

| Comp | Key minions (all in-pool) |
|---|---|
| Beasts - Beetles | Banana Slamma, Turquoise Skitterer, Headhunter Gryphon, Ravaging Scorpid |
| Demons - Self Damage | Ashen Corruptor, Balinda Stonehearth, Eredar Escapist, Devilish Distractor, Brann Bronzebeard |
| Elementals - Unbound Tempest | Unbound Tempest, Kelp Keeper, Brann Bronzebeard, Nomi, Kitchen Nightmare, Tavern Tempest |
| Undead - Attack Scaling | Handless Forsaken, Mummifier, Drustfallen Butcher, Dead Bellringer, Snazzy Phantom |

### A (sample)

Beasts - Tasty Lobstah / Leviathan / Summons · Demons - APM Shop Buff / Shop Buff · Elementals - Stat scaling · Mechs - Magnetics/Spells · Murlocs - Venom Scam / Tidecaller · Undead - APM Undead · Dragons - APM Evoker · Pirates - APM Golden

### B

Murlocs - Family · Quilboar - Bristlemane / Choose One · Mechs - Magnetics / Deathrattle · Dragons - Battlecries

**Dropped:** all Naga comps; any prior comps whose cores were mostly out-of-pool (Monstrous Macaw lines, Young Murk-Eye, Darkgaze Elder, Archlich Kel'Thuzad, etc.).

**Aberration:** tribe B soft prior only — HSReplay had **zero** Aberration comps at fetch time. Soft keys listed in `aberration_soft_key_minions` (real in-pool names only). No invented Aberration fantasy comps.

## Coach wiring

- `hsbg_coach/playstyle_prior.py` — load / tribe_weight / comp_weight / score_board_playstyle / is_out_of_pool / bias_shop_scores
- `tribe_policy.direction_buy_penalty` — hard demote OOP + Naga; soft boost HSReplay S/A keys
- `cards.build_card_kb` — requires `isBattlegroundsPoolMinion`, excludes Naga + duos-exclusive
- Firestone `data/stats/firestone_comp_stats.json` scrubbed of Naga / OOP-core comps

## Login note

HSReplay **comps tier list** is public via page `react_context`. Tier7 perfect-game boards / some analytics endpoints may need auth — not used for this prior refresh.
