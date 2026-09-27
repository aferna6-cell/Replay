# `states.v1`: decision-point states from Firestone replays

One JSONL row per **decision point**: an `Options` block offered to the local
player (`<Player isMainPlayer="true">`) in a Firestone HSReplay XML replay.
Each row carries the `BGTracker.snapshot()` of that moment, the same object the
live Power.log overlay builds, so training and the overlay can share one encoder.

```
python -m hsbg_coach.replay_states --replays data/firestone/raw \
    --manifest data/firestone/raw/pilot_manifest.json --out data/firestone/states/v1
```

Everything under `data/firestone/` is gitignored and must never be committed.

Outputs:

- `<out>/<reviewId>.jsonl.gz`: rows, written only for games that pass every check.
- `<out>/quarantine/<reviewId>.json`: games that fail, with reasons, examples and stats.
- `<out>/report.json`: games processed, pass count and rate, failures grouped by
  reason, rows, bytes and runtime, plus per-game stats (Dark Gift counters and
  atk/health warnings).

Every manifest entry and every `*.xml.gz` in the folder shows up in the report.
`missing_replay`, `not_in_manifest` and `exception` are failure reasons too, so no
game is dropped without a trace. Card names come from a HearthstoneJSON `cards.json`,
downloaded once to `data/firestone/cards.json` (`--cards` overrides the path),
because XML entities carry no names. `--no-opponents` skips per-event opponent
capture, which makes the run about 10x faster, but `opponents_seen` and
`opponent_profiles` are then left empty.

## Pipeline

`hsreplay_xml.iter_events()` streams the (gzip) XML and yields the same `parser.Event`
kinds `parse_line()` produces. `BGTracker.feed()` consumes them unchanged. Tags and
enum values are mapped to the names Power.log prints (`ZONE=PLAY`,
`CARDTYPE=MINION`), using `hs_enums.py`, which is generated from python-hearthstone 9.21.1.
Tags with no name stay numeric, exactly as Power.log prints them. XML quirks:

- **Re-dumps.** `GameEntity` plus the full entity set (same ids) is re-emitted 1-10x
  per game, and dead entities are not re-emitted. Each re-dump emits `RESET_ENTITIES`, and
  `GameState` then drops everything except the Player entities.
- **Player tags inside re-dumps.** A re-dumped `<GameEntity>` has both Player entities'
  tags appended to its own, and no `<Player>` elements are written. They are split back out by
  segment (`CONTROLLER`, `CARDTYPE=PLAYER`, …, `ENTITY_ID`). If they are not split,
  players keep stale gold, and the game `TURN` gets overwritten by a player's `TURN`.
- **Re-sent Options.** The same `Options id` right after a re-dump is one decision
  point. The post-re-dump block is kept.
- `ChangeEntity` is treated as `SHOW_ENTITY` and `HideEntity` as a `ZONE` change. `Block type=ATTACK`
  becomes a `RAW "BLOCK_START BlockType=ATTACK"` event (the phase signal), and an
  `Options` block sets the phase to recruit.

## Row

| field | type | notes |
|---|---|---|
| `schema_version` | `"states.v1"` | |
| `game_id`, `build`, `mmr` | str, int, int | manifest `reviewId`, `buildNumber`, `mmr` |
| `lobby_tribes` | [str] | manifest `tribes.available[].name` (`"MECHANICAL"`, …) |
| `dp_index` | int | 0-based decision point index (re-sent Options counted once) |
| `turn` / `raw_turn` | int | recruit turn `(TURN+1)//2` / GameEntity `TURN` |
| `snapshot` | object | `Snapshot.to_dict()`; rebuild with `Snapshot.from_dict(row["snapshot"])` |
| `hero_power` | `{card_id, name, cost, used, activatable}` or null | hero power in PLAY, including passive ones; `used` = EXHAUSTED; `activatable` = it is a legal option now |
| `dark_discovery` | `{available: bool}` | the `BG36_Button_DarkGift` entity is a legal option now |
| `options` | `[{index, type, entity_id, card_id, zone, targets}]` | legal options only (`error=-1`), where `type` is `POWER` or `END_TURN`. The chosen option is not recorded |

### `snapshot` fields (what replays fill)

| field | from replays | notes |
|---|---|---|
| `game_counter` | filled | always 1 (one game per file) |
| `turn` | filled | raw `TURN`, not the recruit turn |
| `phase` | filled | always `"recruit"` (options are only offered while shopping) |
| `tavern_tier`, `level_cost` | filled | `level_cost` is the tier-up button `COST`, and null at tier 6 |
| `gold` | filled | `RESOURCES - RESOURCES_USED + TEMP_RESOURCES` |
| `hero_health` | filled | effective: `HEALTH + ARMOR - DAMAGE` |
| `hero_armor` | filled | new, optional; null when the ARMOR tag is absent (0) |
| `board`, `hand`, `shop` | filled | `MinionView` rows with the full tag dict; `hand` holds every HAND entity, spells included |
| `MinionView.dark_gift` | filled | new, optional: `{card_id, name}` for `HAS_DARK_GIFT=1` minions, else null |
| `shop_spells` | filled | tavern spells in the shop row (now with `entity_id`) |
| `shop_frozen` | filled | new, optional: any shop minion `FROZEN` |
| `hand_spells` | always empty | live logic wants `CARDTYPE=BATTLEGROUND_SPELL` in HAND, but hand spells are `SPELL` (live Power.log shows the same). Use `hand` |
| `hero_power` | partial | live semantics: null when `COST` is absent, which includes 0-cost powers because zero tags are omitted. Use the row-level `hero_power` |
| `activatable` | filled | Activate-keyword allowlist |
| `dark_gift` | filled | Dark Discovery button with heuristic `usable`. The row-level `dark_discovery` comes from the options |
| `anomaly` | always null | anomalies are not in the current meta |
| `trinkets` | filled | |
| `quests` | always empty (so far) | new, optional: `{card_id, name, progress}` |
| `opponents_seen` | filled | last enemy board seen in combat, captured per event exactly as live |
| `opponent_profiles` | partial | one opponent seat in the log, so the latest opponent only. `tribe` is always null (it is keyed on entity names, which XML lacks) |
| `hero`, `hero_name` | filled | |
| `available_tribes` | partial | live `BACON_SUBSET_*` heuristic only finds `Aberration`. Use `lobby_tribes` |
| `build_tribe`, `notes` | always null / empty | not set by `snapshot()` |

The minion fields from the draft schema live in `MinionView.tags`: tier `TECH_LEVEL`,
golden `PREMIUM`, tribes `CARDRACE` plus the numeric multi-type tags in
`hs_enums.RACE_TAGS`, and keywords `TAUNT`, `DIVINE_SHIELD`, `REBORN`, `WINDFURY`,
`MEGA_WINDFURY`, `VENOMOUS`, `POISONOUS`, `STEALTH`, `MAGNETIC` and `AVENGE`. There is no cleave tag.

### Dark Gift resolution (`BGTracker._minion_dark_gift`)

Only `HAS_DARK_GIFT=1` minions are resolved, in this order:

1. the `DARK_GIFT_ENTITY` card;
2. an attached enchantment in PLAY whose own card or `CREATOR` card starts with
   `BG36_MidGameEffect_000t` (trailing `e\d*` stripped);
3. a Dark Paradox token (`BG36_360t*` or `BG36_360_Gt*`), which is its own gift.

## Fidelity checks (any failure quarantines the game)

1. **Parses:** no exception, and rows = Options blocks − re-sent blocks.
2. **Final board:** the last row's board card ids + golden, in order, match manifest
   `finalComp.board` (`final_board_order` / `final_board_mismatch`). A difference in
   atk/health is only a **warning**. Each warning also lists the stats at the first attack of the
   next combat: `finalComp` is the combat board, so it includes end-of-turn buffs
   applied after the last decision point.
3. **Placement:** final `PLAYER_LEADERBOARD_PLACE` of our hero equals the manifest `placement`.
4. **Invariants on every row:** gold ≥ 0, board ≤ 7, hand ≤ 10, every legal option's
   entity and targets exist in state, and `raw_turn` never decreases.
5. **Dark Gifts:** every `HAS_DARK_GIFT` minion (board/hand/shop) resolves. Every Dark
   Discovery pick whose minion carries `HAS_DARK_GIFT` has its gift on a friendly
   board/hand minion at the next decision point. One-shot gifts (Double Vision:
   "get an extra copy") are spent on pick, so they are counted and not checked.

Tests: `tests/test_hsreplay_xml.py` (synthetic fixture
`tests/fixtures/synthetic_bg_replay.xml`). Set `HSBG_REPLAY_XML=<replay.xml.gz>`, and
optionally `HSBG_REPLAY_MANIFEST=<manifest.json>`, to also run the real-replay test.
