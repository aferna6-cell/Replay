# Battlegrounds visibility & action coverage

Honest map of what the coach can **see** from `Power.log` and what it **advises**
on. Advisory only — no vision, no autoplay, no input injection.

Legend:

| Tag | Meaning |
|---|---|
| **SEEN+ADVISED** | Parsed into Snapshot / ChoiceOffer and ranked by advise/recommend/live |
| **SEEN-only** | Present on Snapshot or ChoiceOffer, but ranking is thin / heuristic / not yet in `legal_actions` |
| **NOT-YET** | Real BG surface; not reliably parsed or not wired into advise |

## Phases

| Phase | Status | Notes |
|---|---|---|
| Hero select | SEEN+ADVISED | `ChoiceParser` → `rank_heroes` (population placement) |
| Trinket select | SEEN+ADVISED | `rank_trinkets` (meta + board fit) |
| Discover / general EntityChoices | SEEN+ADVISED | `rank_discover` (eval net + card2vec when available) |
| Hero-power pick (e.g. Nguyen) | SEEN-only | Classified; `rank_pick` lists options without per-option EV |
| Quest / reward pick | SEEN-only | Same as hero-power pick — needs effect table |
| Recruit | SEEN+ADVISED | Full shop/board/gold/tier when tags calibrate |
| Combat | SEEN+ADVISED | Your board + opponent board → odds; no recruit actions during combat |
| Game over / placement | SEEN+ADVISED | `PLAYER_LEADERBOARD_PLACE` → recorder backfill |

## Recruit actions

| Action | Status | Where |
|---|---|---|
| Buy minion | SEEN+ADVISED | `legal_actions` BUY → advisor / `game_value` |
| Buy tavern spell | SEEN+ADVISED | BUY_SPELL from `shop_spells` |
| Sell minion | SEEN+ADVISED | SELL |
| Roll shop | SEEN+ADVISED | ROLL |
| Tier up | SEEN+ADVISED | LEVEL (`level_cost` when tagged) |
| Freeze shop | SEEN+ADVISED | FREEZE when `shop_frozen` is false |
| Unfreeze shop | SEEN+ADVISED | UNFREEZE when any shop minion has `FROZEN=1` |
| Hero power | SEEN+ADVISED | When `hero_power.usable` |
| Reposition | SEEN+ADVISED | Computed; live overlay often skips heavy sim for latency |
| End turn | SEEN+ADVISED | END (usually buried vs concrete moves) |
| Play minion from hand | SEEN+ADVISED | PLAY in `legal_actions` + overlay hand lines |
| Cast hand spell | SEEN+ADVISED | PLAY_SPELL + `spell_target` lines |
| Choose-One battlecry half | SEEN-only | Curated table in `choose_one.py` (few cards); else generic |
| Magnetize onto host | SEEN+ADVISED | Overlay / magnetize helper when Magnetic in hand |

## Lobby / combat boards

| Signal | Status | Notes |
|---|---|---|
| Your board | SEEN+ADVISED | Snapshot.board |
| Shop minions | SEEN+ADVISED | Snapshot.shop (`# CALIBRATE` shop zone) |
| Last opponent combat board | SEEN+ADVISED | `opponents_seen` / odds panel |
| Lobby opponent profiles | SEEN+ADVISED | heroes / tribes / strength → insights |
| Equipped trinkets | SEEN+ADVISED | Snapshot.trinkets (comp lean) |
| Anomaly | SEEN+ADVISED | Snapshot.anomaly → buy bias when known |
| Buddy offers / buddy board | NOT-YET | No dedicated parser path yet |
| Quest progress UI | NOT-YET | Quest *picks* are SEEN-only; progress meters not tracked |
| Gold / tier / HP | SEEN+ADVISED | `# CALIBRATE` markers remain on some tags |

## Recording & learning

| Piece | Status |
|---|---|
| `(state, action, placement)` JSONL under `data/game-*.jsonl` | SEEN+ADVISED path — every `watch` game |
| Continual background retrain | Wired (`continual.BackgroundTrainer`) |
| Session retrain CLI | `python -m hsbg_coach learn` / `./scripts/retrain.sh` |
| Adaptive population↔personal mix | Real (`config.personal_weight` → train upsample) |
| Live hot-swap of `ml/eval_net.pt` | mtime check in `LiveCoach._maybe_reload_scorer` |

## Known gaps (follow-on PRs)

1. **Buddy** lifecycle (offer, play, death) — not in Snapshot yet.
2. **Quest progress** / reward EV table — picks are listed, not ranked by effect.
3. **Hero-power pick EV** — same honesty as quests.
4. **Per-click buy/sell action labels** in the recorder — currently end-of-recruit `END_TURN` snapshots (placement still backfilled). Finer action attribution needs log click/packet calibration.
5. Shop zone / gold / placement tags still carry `# CALIBRATE` in `bg.py`.

Do not treat this doc as a promise of full client fidelity — it is a coverage contract for what Power.log gives us today.

## Season 14 critical actions (live-tested)

| Action | Status | Detection |
|---|---|---|
| Hero power (clickable) | SEEN+ADVISED | Snapshot.hero_power (COST present) + Options error=NONE |
| Dark Gift button | SEEN+ADVISED | Snapshot.dark_gift + Options name/cardId patterns |
| Activate minion | SEEN+ADVISED | HAS_ACTIVATE_POWER + BACON_TRIGGER_XY + Options |
| Buy tavern spell | SEEN+ADVISED | shop_spells; spell_roles scoring boosted for good/cheap |
| Single next-move UX | SEEN+ADVISED | `advice_lines(top=1)` + `format_next` / overlay NEXT line |
| Recompute on act | SEEN+ADVISED | cache key includes HP/activate/gift/options fingerprint |

Passive hero powers (BACON_TRIGGER_UPBEAT, no COST) are intentionally **not** advised as clicks.

## Meta playstyle (scoring priors)

| Signal | Status | How it affects the #1 move |
|---|---|---|
| Firestone/HSReplay listed comps | SEEN+ADVISED | `meta_strategy.is_listed_comp_piece` promotes core cards |
| Tribe avg placement (first-place style) | SEEN+ADVISED | `tribe_placement_table` from comp avgPosition×popularity |
| Lobby opponent tribes | SEEN+ADVISED | Soft boost for tribes seen in `opponent_profiles` |
| Hero best tribes | SEEN+ADVISED | `build_hero_context` + meta prior |
| Forced lock to one tribe | NOT done | Soft only — listed-comp pivot when shop opens another viable tribe |

Implementation: `hsbg_coach/meta_strategy.py` → `_meta_strategy_adjust` in `game_value.rank_actions`.
