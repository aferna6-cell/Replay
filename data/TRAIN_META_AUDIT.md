# TRAIN META AUDIT — 36.6.1 (2026-09-23T18:49:28Z)

Locked decisions: approved synth v2; train on intact comps + synths + scrubbed thin150; also heroes/trinkets/spells as support; T7 useful but NOT required; no Firestone; no VODs; Naga out.

## 1. Live pool / Naga

- Snapshot minions: **247** (all `in_pool=true`)
- Naga minions in snapshot: **0** (none)
- Live tribes (10): aberration, beast, demon, dragon, elemental, mech, murloc, pirate, quilboar, undead
- Excluded: naga (meta.excluded_tribes + comps train_weight=0)
- tribes.json out_of_pool still on HSReplay: [{'id': 'naga', 'tribe_id': 92, 'name': 'Naga', 'first_place_rate': 11.2, 'avg_placement': 3.94, 'popularity': 10.2, 'placement_distribution': {'1': 16.42, '2': 14.59, '3': 13.9, '4': 12.85, '5': 12.1, '6': 11.01, '7': 10.44, '8': 8.71}, 'typical_final_board': ['felfire conjurer', 'seafloor recruiter', 'fauna whisperer', 'gatekeeper amalgam', 'torrential ruiner', 'drakkari enchanter', 'balinda stonehearth'], 'hsreplay_url': 'https://hsreplay.net/battlegrounds/compositions/43/naga?mmrPercentile=TOP_50_PERCENT&timeRange=LAST_7_DAYS', 'stats_filters': {'mmr_range': 'Top 50% (5,800+)', 'time_frame': 'Last 7 days', 'games': '540,000'}, 'source': 'hsreplay_compositions_page_visible', 'lobby_strength': None, 'lobby_win_rate': 11.2, 'top_comp_ids': [], 'notes': 'Still listed on HSReplay compositions page; excluded from live pool filter in minions.json', 'in_current_lobby_pool': False}]

## 2. Kept intact snapshot comps (11)

Every core/key/enabler/addon id verified in minions.json in-pool.

- **comp_89** Pirates - APM Golden (pirate, rank A): all 10 pieces in-pool OK
- **comp_76** Demons - APM Shop Buff (demon, rank A): all 15 pieces in-pool OK
- **comp_3** Undead - APM Undead (undead, rank A): all 15 pieces in-pool OK
- **comp_14** Undead - Attack Scaling (undead, rank S): all 17 pieces in-pool OK
- **comp_38** Beasts - Beetles (beast, rank S): all 14 pieces in-pool OK
- **comp_13** Demons - Self Damage (demon, rank S): all 18 pieces in-pool OK
- **comp_41** Demons - Shop Buff (demon, rank A): all 15 pieces in-pool OK
- **comp_12** Beasts - Summons (beast, rank A): all 16 pieces in-pool OK
- **comp_87** Beasts - Tasty Lobstah (beast, rank A): all 13 pieces in-pool OK
- **comp_67** Murlocs - Tidecaller (murloc, rank A): all 15 pieces in-pool OK
- **comp_8** Murlocs - Venom Scam (murloc, rank A): all 10 pieces in-pool OK

## 3. Dropped / zero-weight (13)

- **comp_73** Dragons - APM Evoker (dragon): missing ['BG32_822', 'BG24_004', 'BG29_810']
- **comp_15** Dragons - Battlecries (dragon): missing ['BG24_004', 'BG32_822', 'BG29_810']
- **comp_90** Quilboar - Bristlemane (quilboar): missing ['BG36_510']
- **comp_88** Quilboar - Choose One (quilboar): missing ['BG36_510']
- **comp_68** Mechs - Deathrattle (mech): missing ['BGS_012']
- **comp_55** Nagas - End Of Turn/Spell Buff (naga): missing ['BG32_837', 'BG36_622', 'BG32_835']
- **comp_93** Murlocs - Family (murloc): missing ['BG35_142', 'BG35_141']
- **comp_20** Nagas - Groundbreaker (naga): missing ['BG31_035', 'BG34_925', 'BG23_008', 'BG31_920']
- **comp_83** Beasts - Leviathan (beast): missing ['BG23_008']
- **comp_2** Mechs - Magnetics (mech): missing ['BG26_148', 'BG36_760', 'BG34_925']
- **comp_91** Mechs - Magnetics/Spells (mech): missing ['BG36_760']
- **comp_37** Elementals - Stat scaling (elemental): missing ['BG32_846', 'BG36_351', 'BG32_842', 'BG31_843', 'BGS_127']
- **comp_92** Elementals - Unbound Tempest (elemental): missing ['BGS_104', 'BG32_846']

## 4. Synth comps v2 (5)

Source: `data/train_synth_comps_v2.json` (version v2_playable_no_t7).
All pieces in-pool; **no T7 on cores/keys/example boards** (tier<=6); display names match minions.json (0 mismatches).
T7 policy note written into each synth (`t7_note`, notes/playbook).

- **synth_aberration_1** Aberrations - Discard Engine (aberration): cores/keys/board tier<=6, all pieces in-pool, names match minions.json
- **synth_dragon_1** Dragons - Rally Poet (dragon): cores/keys/board tier<=6, all pieces in-pool, names match minions.json
- **synth_elemental_1** Elementals - Unbound Tempest (elemental): cores/keys/board tier<=6, all pieces in-pool, names match minions.json
- **synth_mech_1** Mechs - Glambot Magnetics (mech): cores/keys/board tier<=6, all pieces in-pool, names match minions.json
- **synth_quilboar_1** Quilboar - Choose One Gems (quilboar): cores/keys/board tier<=6, all pieces in-pool, names match minions.json

Tribes covered by synth: aberration, dragon, elemental, mech, quilboar.

## 5. Thin150 scrub

| | count |
|--|--:|
| Source boards | 150 |
| **Clean (every card_id in-pool)** | **72** |
| Dirty excluded | 78 |

Clean copy: `data/train_perfect_thin150_clean/expert_perfect_thin150_clean.jsonl`

## 6. Tribe coverage (kept + synth active train comps)

| Tribe | # comps |
|--|--:|
| aberration | 1 |
| beast | 3 |
| demon | 3 |
| dragon | 1 |
| elemental | 1 |
| mech | 1 |
| murloc | 2 |
| pirate | 1 |
| quilboar | 1 |
| undead | 2 |

All 10 live tribes have ≥1 train comp: **YES**

## 7. Heroes / trinkets / spells

| File | Records | Notes |
|--|--:|--|
| heroes.json | 118 | 118 with avg_placement; 118 with leveling_curve; no in_pool flag |
| trinkets.json | 293 | 255 with avg_placement; no explicit out-of-pool flag |
| spells.json | 78 | in_pool true=72, false=6; coach takes on 78 |

Stale/out-of-pool: **6 spells** flagged `in_pool=false` (excluded from support). Heroes/trinkets lack pool flags; support builder only uses those related to **active train comps** or favorable live tribes.

## 8. Wiring actions taken

1. `comps.json` — kept 11 at train_weight=highest; dropped/naga at 0; appended 5 synth v2; T7 notes on active comps.
2. `meta.json` — train_weights + training_notes (T7 optional, no Firestone/VODs, thin150 scrubbed).
3. `train_synth_comps_v2.json` — explicit `t7_note` / `t7_required=false`.
4. `data/train_perfect_thin150_clean/` — scrubbed trajectories.
5. `ml/hsreplay_snapshot_dataset.py` — heroes/trinkets/spells SUPPORT examples.
6. Docs: `docs/TRAIN_HSREPLAY_LOCAL.md`, `data/TRAIN_LOCAL_COMMANDS.md`.

## 9. Risks

- Synth comps are playable reconstructions, not HSReplay-measured placements (labels from rank/label_placement).
- Thin150 clean is only 72 boards; some are sparse early boards.
- Hero/trinket related_comp_ids may point at dropped comps — support builder skips those links.
- T7 minions still exist in minions.json (useful adds); model must not overfit them as mandatory.

## 10. Train result (this session)

- Completed 40 epochs locally.
- val MAE **0.617**, Pearson r **0.805**
- Checkpoint: `results/eval_net_hsreplay_36_6_1.pt` → `ml/eval_net.pt`
