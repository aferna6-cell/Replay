# `states.v2`: decision-point states from Firestone replays

One JSONL row per **decision point**, in game order, for the local player
(`<Player isMainPlayer="true">`) in a Firestone HSReplay XML replay:

- `kind="options"`: an `Options` block (shopping actions);
- `kind="choice"`: a `Choices` block offered to the local player. This covers the
  hero pick, discovers and triple rewards, trinkets, Dark Discovery, Sire quests, etc.

Each row carries the `BGTracker.snapshot()` of that moment, the same object the
live Power.log overlay builds, so training and the overlay can share one encoder.

**Entity ids are only stable within one snapshot.** Minions are re-created
each combat, and on buy / triple, and re-dumps reset state. So the same card has
different `entity_id`s across rows. Match on `card_id` across rows, and on
`entity_id` only within a row (options, choice cards, ChosenEntities).

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
- **Choices.** `<Choices>` becomes a `CHOICES` event (`type` mapped through
  `ChoiceType`: `MULLIGAN` / `GENERAL`) and `<ChosenEntities>` becomes `CHOSEN`. Only blocks whose
  `playerID` equals the main `<Player>`'s entity id become rows. Every such block
  becomes one row, including an identical offer repeated before any pick (counted as
  `choices_repeated`).
- `ChangeEntity` is treated as `SHOW_ENTITY` and `HideEntity` as a `ZONE` change. `Block type=ATTACK`
  becomes a `RAW "BLOCK_START BlockType=ATTACK"` event (the phase signal), and an
  `Options` block sets the phase to recruit.

## Row

| field | type | notes |
|---|---|---|
| `schema_version` | `"states.v2"` | see the changelog below |
| `game_id`, `build`, `mmr` | str, int, int | manifest `reviewId`, `buildNumber`, `mmr` |
| `lobby_tribes` | [str] | manifest `tribes.available[].name` (`"MECHANICAL"`, …) |
| `kind` | `"options"` or `"choice"` | |
| `dp_index` | int | 0-based, chronological over both kinds (re-sent Options counted once) |
| `options_index` | int | options rows only: 0-based index among options rows, i.e. the states.v1 `dp_index` |
| `turn` / `raw_turn` | int | recruit turn `(TURN+1)//2` / GameEntity `TURN` |
| `snapshot` | object | `Snapshot.to_dict()`; rebuild with `Snapshot.from_dict(row["snapshot"])` |
| `hero_power` | `{card_id, name, cost, used, activatable}` or null | hero power in PLAY, including passive ones; `used` = EXHAUSTED; `activatable` = it is a legal option now |
| `dark_discovery` | `{available: bool}` | the `BG36_Button_DarkGift` entity is a legal option now |
| `options` | `[{index, type, entity_id, card_id, zone, targets, sub_options}]` | legal options only (`error=-1`), where `type` is `POWER` or `END_TURN`. The chosen option is not recorded |
| `options[].sub_options` | `[{index, entity_id, card_id, targets}]` | v2: Choose One variants (`<SubOption>` children), legal ones only (`error=-1`, the default when the attribute is absent). `card_id` is resolved through the tracker at row time. Usually empty |

**Presence in `options` is the playable / activatable flag.** The game only offers
what can be done right now, so:

- a hand card is playable iff its `entity_id` is in `options`;
- a board minion's Activate is available iff it is in `options`;
- the hero power is usable iff it is in `options` (the same as `hero_power.activatable`);
- a shop minion or spell is buyable iff it (or its buy handle) is in `options`.

Nothing else in the row carries a separate "playable" bit. `options[].targets` lists
the legal targets, and `sub_options[].targets` lists them per Choose One variant.

`hero_power`, `dark_discovery` and `options` are only on options rows. Choice rows
have `choice` instead:

| `choice` field | notes |
|---|---|
| `choice_id` | the Choices `id` attribute (usually 0) |
| `choice_type` | `MULLIGAN` (the hero pick) or `GENERAL` |
| `choice_kind` | `hero` (the MULLIGAN hero pick only), `dark_discovery` (source `BG36_MidGameEffect_010`), `quest` (only `QUEST=1` cards), `trinket` (only `BATTLEGROUND_TRINKET`), `discover` (only minions / spells: triple rewards, discover effects), `other` (e.g. hero-power offers, Friendly Wager (TB_BaconShop_HP_081) combat guesses) |
| `source_entity_id`, `source_card_id`, `source_name` | the Choices `source` entity (e.g. `TB_BaconShop_Triples_01`, `BG30_Trinket_1st`, `BG24_QuestsPlayerEnch_t`). For the hero pick it has no card id |
| `min`, `max` | how many cards may be picked |
| `cards` | `[{entity_id, card_id, name, cardtype, tags, dark_gift}]` in offer order. `tags` is the full tag dict. `dark_gift` is `{card_id, name}` via `HAS_DARK_GIFT` or `DARK_GIFT_ENTITY` (Dark Discovery offers), else null |

The pick is **not** recorded. Match `cards[].entity_id` against the next
`ChosenEntities` (same snapshot, so the ids line up).

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
| `MinionView.buy_cost` | filled (shop) | v1.1, optional: `COST` of the slot's buy handle (`TB_BaconShop_DragBuy`; unnamed tag 2442 = the shop entity). Missing COST = 0. Null outside the shop |
| `reroll_cost` | filled | v1.1, optional: `COST` of `TB_BaconShop_8p_Reroll_Button` in PLAY (missing COST = 0, a free refresh) |
| `shop_spells` | filled | `{name, card_id, cost, entity_id, buy_cost}`, sorted by shop position. Tavern spells (`BATTLEGROUND_SPELL` or `SPELL`) in PLAY under the shop controller. The controller is Bartender Bob's (`TB_BaconShopBob*`) or any shop minion's. Before v1.1, a shop with no minions yielded no spells |
| `shop_frozen` | filled | new, optional: any shop minion `FROZEN` |
| `hand_spells` | filled | our `SPELL` / `BATTLEGROUND_SPELL` cards in HAND. A missing COST = 0. Before v1.1 this was always empty, because in-hand spells are `SPELL` and 0-cost ones have no COST tag |
| `hero_power` | filled | our hero power in PLAY. A missing COST = 0. Null for passive powers (`HIDE_COST=1`) and when no hero power is in PLAY (e.g. Sire after the hero-power quest completes). Before v1.1, 0-cost powers were null, and stale non-PLAY powers were picked. The row-level `hero_power` also covers passives |
| `activatable` | filled | Activate-keyword allowlist |
| `dark_gift` | filled | Dark Discovery button with heuristic `usable`. The row-level `dark_discovery` comes from the options |
| `anomaly` | always null | anomalies are not in the current meta |
| `trinkets` | filled | |
| `quests` | filled (Sire only) | `[{entity_id, card_id, name, progress, goal, reward_card_id, source_card_id, completed}]`, sorted by `entity_id`; see below. Empty for every other hero |
| `opponents_seen` | filled | last enemy board seen in combat, captured per event exactly as live |
| `opponent_profiles` | partial | one opponent seat in the log, so the latest opponent only. `tribe` is always null (it is keyed on entity names, which XML lacks) |
| `hero`, `hero_name` | filled | |
| `available_tribes` | partial | live `BACON_SUBSET_*` heuristic only finds `Aberration`. Use `lobby_tribes` |
| `build_tribe`, `notes` | always null / empty | not set by `snapshot()` |

`MinionView.buy_cost`, `shop_spells[].buy_cost`, `reroll_cost` and `hero_power` are
computed in `BGTracker.snapshot()` (`hsbg_coach/bg.py`) from entity tags only: buy-handle
`COST` and tag 2442, reroll-button `COST`, and hero power `ZONE` / `COST` / `HIDE_COST`.
They never read Options, so the live Power.log path fills them identically. The
row-level `hero_power.activatable`, `dark_discovery.available` and `options` are the only
fields taken from the Options block.

The minion fields from the draft schema live in `MinionView.tags`: tier `TECH_LEVEL`,
golden `PREMIUM`, tribes `CARDRACE` plus the numeric multi-type tags in
`hs_enums.RACE_TAGS`, and keywords `TAUNT`, `DIVINE_SHIELD`, `REBORN`, `WINDFURY`,
`MEGA_WINDFURY`, `VENOMOUS`, `POISONOUS`, `STEALTH`, `MAGNETIC` and `AVENGE`. There is no cleave tag.

### Dark Gift resolution (`BGTracker._minion_dark_gift`)

Only `HAS_DARK_GIFT=1` minions are resolved, in this order:

1. the `DARK_GIFT_ENTITY` card;
2. an attached enchantment in PLAY whose own card or `CREATOR` card starts with
   `BG36_MidGameEffect_000t`. A trailing `e` / `e2` is the spell's enchantment;
   `te` / `te2` is Persistent Poet's permanent copy of that enchantment
   ("Adjacent Dragons permanently keep Bonus Keywords and stats gained in
   combat"). Both name the same spell (`...000t64te` → `...000t64`);
3. a Dark Paradox token (`BG36_360t*` or `BG36_360_Gt*`), which is its own gift;
4. a gift remembered when its enchantment was attached, including one inherited
   across `COPIED_FROM_ENTITY_ID`. Timewarped Radio Star
   (`BG34_Giant_330`, "Get a copy of the enemy minion that killed this with
   full Health and enchantments") copies after death has already removed the
   killer's enchantments. The hand copy keeps `HAS_DARK_GIFT` and never
   receives a gift enchantment of its own (b138f295: Persistent Poet entity
   9052, copied from 9051 copied from 8958, whose enchantment was
   `BG36_MidGameEffect_000t64te`). The intermediate copy's `COPIED_FROM` is
   cleared before the next decision point, so the gift is recorded when the
   tag is set. A `HAS_DARK_GIFT` minion with no enchantment and no copied
   gift still fails `dark_gift_unresolved`.

A re-dump drops entities that died before it (see Pipeline), including the gift
spell itself, while `DARK_GIFT_ENTITY` and the enchantment's `CREATOR` still point
at it. `GameState` therefore keeps the card id of every entity a re-dump drops
(`dropped_card_ids`, read through `GameState.card_id_of`); ids are never reused
within a game. The Harpy's Talons gift (`BG36_MidGameEffect_000t13`) is the usual
case: its enchantment is `EDR_100t13e`, which does not carry the gift prefix.
When that enchantment's CREATOR was never in the replay at all, `EDR_100t13e` itself
maps to `BG36_MidGameEffect_000t13` (`bg.DARK_GIFT_ENCHANTMENTS`). It has the gift's
name and text ("Harpy's Talons: Divine Shield, Windfury"), and in the 1,000-game corpus
the only gift spell that ever creates it is `BG36_MidGameEffect_000t13` (1,307 times;
other copies name a minion as CREATOR). It is only used for `HAS_DARK_GIFT` minions.

A golden made from gifted copies carries every copy's gift enchantment (about 1.6% of
gifted board/hand snapshots in a 100-game sample). `dark_gift` names only the first one
found; listing all of them would need a schema change.

### Quests (`BGTracker._quests`)

Quests come from Sire Denathrius (`BG24_HERO_100` and its skins `BG24_HERO_100_SKIN_*`,
e.g. `_SKIN_A` "Sire Melodious", `_SKIN_E` "Boss Denathrius", matched by
`bg.SIRE_HERO_RE`) and from his buddy Shady Aristocrat (`BG24_HERO_100_Buddy`,
"When you sell this, Discover a Quest"). Any hero can end up with the buddy: in
5ec28cd6 (The Rat King) and f70e9184 (Kurtrus Ashfallen) the trinket Wisdomball
Supply (`BG31_MagicItem_903`) gave Knockoff Wisdomball (`BG30_802`), whose refresh
created Shady Aristocrat; the player sold it and picked a quest. In f70e9184 Kurtrus's
hero power Glaive Ricochet (`BG20_HERO_280p5`) then made a plain copy of the buddy,
which was sold for a second quest. What the replays show:

- **Offer:** the hero power `BG24_HERO_100p` "Whodunit?" creates the enchantment
  `BG24_QuestsPlayerEnch_t`. That enchantment creates 2 quest + reward pairs in
  SETASIDE and a `Choices` (`source` = the enchantment) over the 2 quests. The Sire
  buddy `BG24_HERO_100_Buddy` offers 3 quests the same way, each rewarding a
  `BG24_HERO_100_Buddyt` "Coin Pouch". A quest is `CARDTYPE=SPELL` with `QUEST=1` and
  `CREATOR` = the source. Its `TAG_SCRIPT_DATA_ENT_1` points at its reward entity, and the
  reward points back through tag 1396 (not named in `hs_enums`). The reward is a
  `BATTLEGROUND_QUEST_REWARD` for hero-power quests and a `SPELL` for buddy quests.
  `QUEST_REWARD_DATABASE_ID` holds the reward dbfId.
- **Active:** the picked quest moves to **ZONE=SECRET**. `QUEST_PROGRESS` counts up
  toward `QUEST_PROGRESS_TOTAL` (the goal is set when the quest is offered and can
  differ from the card default). The unpicked quest goes to REMOVEDFROMGAME.
- **Completion:** progress reaches the goal. The quest goes SECRET → SETASIDE →
  REMOVEDFROMGAME, and its tags reset to the card defaults (`QUEST_PROGRESS` 0,
  `QUEST_PROGRESS_TOTAL` = the card's base goal). The hero gets `BACON_QUEST_COMPLETED=1`.
  For hero-power quests, a new `BATTLEGROUND_QUEST_REWARD` entity with `CREATOR` = the
  quest appears in PLAY. Rewards such as "Perpetual Incantation" (`BG33_Reward_020`)
  re-create the quest as a new entity in SECRET, which has no reward pointer.

`quests` lists (a) every quest of ours in SECRET with `QUEST=1` (`completed=false`,
`progress`/`goal` from the tags, `reward_card_id` via `TAG_SCRIPT_DATA_ENT_1`,
`source_card_id` = CREATOR card) and (b) every `BATTLEGROUND_QUEST_REWARD` of ours
in PLAY whose CREATOR is a quest. That entry uses the quest's `entity_id`/`card_id`,
with `completed=true` and `progress`/`goal` null, because the reset tags no longer
carry them. Buddy quests and re-created quests have no reward in PLAY, so they drop out
of the list when completed. A quest that completes during the same block it was picked
in never appears at a decision point.

## Fidelity checks (any failure quarantines the game)

1. **Parses:** no exception, and options rows = Options blocks − re-sent blocks.
2. **Final board:** Firestone's `finalComp` is the start-of-combat board of turn
   `finalComp.turn`, which is often the turn before the game's last turn. The board
   card ids + golden, in order, of our last options row of that turn (the last row if
   the turn is missing) must match `finalComp.board`. If the player changed the board
   after that decision point, the board at that turn's first attack may match instead
   (stat `final_comp_from_combat`; stat `final_comp_earlier_turn` when the turn is not
   our last). Otherwise: `manifest_final_comp_suspect` when the manifest board equals
   our board at some other turn (the manifest's turn label is off), else
   `final_board_order` / `final_board_mismatch`. A difference in atk/health is only a
   **warning**, listing the stats at the first attack when that board matched.
3. **Placement:** final `PLAYER_LEADERBOARD_PLACE` of our hero equals the manifest
   `placement`. When it does not, and our player's final `PLAYSTATE` is `WON` with our
   place 1, the reason is `manifest_placement_suspect` (the manifest scan followed the
   player's `HERO_ENTITY` to a temporary hero and kept a stale place) instead of
   `placement_mismatch`.
4. **Invariants on every options row:** gold ≥ 0 (gold is null until the player's
   `RESOURCES` tag is first set, at the first 1–4 decision points of some games; that
   is counted as `gold_unknown_rows`, and null after gold was known fails), board ≤ 7,
   hand ≤ 10, every legal option's
   (and sub-option's) entity and targets exist in state, and `raw_turn` never decreases.
5. **Dark Gifts:** every `HAS_DARK_GIFT` minion (board/hand/shop) resolves. A Dark
   Discovery pick whose minion carries `HAS_DARK_GIFT` and is still in our hand or on
   our board at the next decision point must carry the picked gift. A pick already
   gone by then (sold, or tripled: the golden carries every copy's gift enchantment,
   but `dark_gift` names only the first one found) is verified if its gift is on another
   held minion, otherwise counted as
   `dark_discovery_picks_gone`, not failed. When the offered minion has
   `HAS_DARK_GIFT` but no `DARK_GIFT_ENTITY` (stat `dark_discovery_picks_gift_unnamed`),
   the held pick only has to carry some gift. A pick with neither fails
   `dark_discovery_pick_without_gift`. One-shot gifts (Double Vision:
   "get an extra copy") are spent on pick, so they are counted and not checked.
6. **Choices:** every main-player `Choices` block yields exactly one choice row
   (`choice_row_count_mismatch`). Each offers ≥ 1 card (`choice_no_cards`) and every
   card entity exists with a card id (`choice_card_missing`). The turn invariant also
   covers choice rows. A game with no hero-pick row gets the **warning**
   `hero_pick_missing` (per-game `game_warnings`, report `game_warnings_by_reason`),
   which does not quarantine the game.
7. **Quests:** a row whose `hero` is not Sire (`SIRE_HERO_RE`) may only hold quests
   whose `source_card_id` is Shady Aristocrat (`BG24_HERO_100_Buddy*`), otherwise
   `quests_on_non_sire_hero`; stat `quest_snapshots_non_sire`. For each quest `entity_id` across rows:
   progress never decreases (`quest_progress_decreased`), except Pressure the
   Authorities (`BG27_Quest_801`, "Get your warband to {0} total Attack"),
   whose `QUEST_PROGRESS` is the live sum of board attack and may fall when
   that sum falls — a drop is accepted only when the new progress equals the
   board's total attack (a8aebfa4: progress 9 → 0 at dp 21 while the board was
   empty, then 11 once a minion was back). The goal is `QUEST_PROGRESS_TOTAL`
   (20 in that game), not the card's baseline script value (28, tags 2/3 and
   the initial 535 before the offer overwrites it). The goal is present
   (`quest_goal_missing`), progress ≤ goal while not completed
   (`quest_progress_over_goal`), and a completed quest never reverts to active
   (`quest_uncompleted`). A Sire game with no quest at any decision point fails
   `sire_quests_missing`. Stats: `quests_seen`, `quests_completed` (quests seen
   with `completed=true`), `quest_snapshots`, and `quest_goal_changed` (counted
   only).

Tests: `tests/test_hsreplay_xml.py` (synthetic fixture
`tests/fixtures/synthetic_bg_replay.xml`). Set `HSBG_REPLAY_XML=<replay.xml.gz>`, and
optionally `HSBG_REPLAY_MANIFEST=<manifest.json>`, to also run the real-replay test.

## Changelog

- **states.v2 checks, two quarantined 36.6.3 wins** (the row schema is unchanged)
  - Dark Gifts: remember a gift enchantment (including Poet `te` / `te2`
    permanent copies) and inherit it across `COPIED_FROM_ENTITY_ID`, so a
    Radio Star hand copy whose enchantments were removed before the copy
    still names the source gift.
  - Quests: Pressure the Authorities may lose progress when that loss equals
    the board's current total attack. Cumulative quests still fail
    `quest_progress_decreased`.

- **states.v2 checks, after the 1,000-game run** (the row schema is unchanged)
  - Card ids of entities dropped by a re-dump are kept, so Dark Gifts and quest
    `source_card_id` resolve through them.
  - Final board: compared at `finalComp.turn`, with the start-of-combat board as a
    fallback. New reason: `manifest_final_comp_suspect`.
  - Placement: new reason `manifest_placement_suspect`.
  - Gold: null before the first `RESOURCES` is unknown, not a failure.
  - Dark Discovery picks: only checked while the picked minion is still held; an
    offer without `DARK_GIFT_ENTITY` is checked for any gift.
  - `EDR_100t13e` resolves to Harpy's Talons when its CREATOR is unknown.
  - Quests: a non-Sire hero may hold quests from Shady Aristocrat.

- **states.v2**
  - Adds choice rows (`kind`, `choice`) and a chronological `dp_index`. The old Options-only counter is kept as `options_index`.
  - `options[].sub_options`: legal Choose One variants.
  - Options rows are otherwise unchanged from v1.1.

- **states.v1.1**
  - `shop_spells` accepts `SPELL` as well as `BATTLEGROUND_SPELL` and is anchored on Bartender Bob's controller, so a shop of only spells is captured.
  - `hand_spells` is filled.
  - `Snapshot.hero_power` treats a missing COST as 0 and only considers the power in PLAY. This also changes the live overlay.
  - New optional fields: `MinionView.buy_cost`, `shop_spells[].buy_cost`, `Snapshot.reroll_cost`.
