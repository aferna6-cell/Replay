# `labels.v1`: decision-point training rows

`hsbg_coach/replay_labels.py` turns State Builder's `states.v1` rows
(`docs/replay_states_v1.md`) into weighted behaviour-cloning rows:
`(state, every legal option, chosen option)`.

```
python -m hsbg_coach.replay_labels --states data/firestone/states/v1 \
    --replays data/firestone/raw --manifest data/firestone/raw/pilot_manifest.json \
    --corpus-manifest data/firestone/raw/manifest.json --out data/firestone/labels/v1
```

Everything under `data/firestone/` is gitignored and must never be committed.

Inputs: the states folder (`<reviewId>.jsonl.gz`), the replay folder
(`<reviewId>.xml.gz`, read again for the player's actions), the manifest of games
to label (placement, MMR) and the corpus manifest (MMR percentiles for weights;
defaults to `--manifest`).

Outputs:

- `<out>/<reviewId>.jsonl.gz`: labeled rows (one per decision point that could be labeled).
- `<out>/quarantine/<reviewId>.jsonl`: decision points that could not be labeled, with a reason.
- `<out>/choices/<reviewId>.jsonl`: Choices picks (hero / trinket / quest / discover). See below.
- `<out>/summary.json`: every check and count listed under "Checks".

## Row

One row per State Builder decision point (`dp_index`, i.e. one `Options` block).

| field | type | notes |
|---|---|---|
| `game_id`, `build`, `mmr` | str, int, int | copied from the state row |
| `current_patch` | bool | `build == 253216` |
| `turn`, `dp_index` | int | copied from the state row |
| `placement` | int | manifest placement; always 1 (other games are dropped) |
| `state` | object | the state row's `snapshot` (`Snapshot.to_dict()`), unchanged |
| `options` | [option] | the server-legal options at this decision point, expanded (see below) |
| `chosen` | int | index into `options`; always valid (the run raises otherwise) |
| `inference_method` | str | see "Inference" |
| `confidence` | `high` / `medium` / `low` | see "Inference"; v1 emits no `low` rows |
| `weight` | float | MMR weight, see "Weight" |

Row-level state fields that are not in the snapshot (`lobby_tribes`, row-level
`hero_power`, `dark_discovery`, `raw_turn`) are not copied; join the states file
on `(game_id, dp_index)` if needed.

### Option

```json
{"type": "buy|sell|reroll|freeze|level|hero_power|play|discover|end_turn|reposition",
 "card_id": "str|null",
 "source": {"zone": "board|shop|hand|none", "slot": "int|null"},
 "target": {"zone": "board|shop|hand|none", "slot": "int|null"},
 "position": "int|null",
 "src_entity": "int|null", "target_entity": "int|null",
 "choice_kind": "str|null"}
```

- **Slots** are 0-based indexes into the snapshot lists (`board`, `hand`, `shop`).
  State Builder sorts those lists by `ZONE_POSITION` (checked on all 2,361 pilot
  states: board positions are always 1..n). **Shop spells** use slot
  `len(shop) + j` (j = index into `shop_spells`; spells carry no position in states.v1).
- **`position`**: for `play` of a hand minion, the insertion index `0..len(board)`
  (= ZONE_POSITION - 1 after the play); for `reposition`, the final index (never
  equal to the source slot). Null otherwise.
- **`src_entity` / `target_entity`**: raw XML entity ids, debug only. For buy/sell,
  `src_entity` is the card bought/sold (the DragBuy/DragSell helper id is dropped).
- **`card_id`**: the card acted on (the shop card for buy, the board minion for sell,
  the Choose One variant for a Choose One play, the hero power for hero_power).
  Null for `reroll`, `freeze`, `level` and `end_turn`.
- `choice_kind` is null except for `discover` options (none in v1 rows, see Choices).

### Option enumeration

Only server-legal options (`error=-1`, as kept by State Builder) are used. Each is
expanded into one labeler option per distinct target and board position, then
de-duplicated on `(type, card_id, source, target, position)` (plus `target_entity`
when the target is outside board/shop/hand, e.g. our hero).

| server option entity | labeler options |
|---|---|
| `END_TURN` | `end_turn` |
| `TB_BaconShop_DragBuy` / `_DragBuy_Spell` | `buy`, one per target in the shop (minions and spells) |
| `TB_BaconShop_DragSell` | `sell`, one per target on the board |
| `TB_BaconShop_8p_Reroll_Button` | `reroll` |
| `TB_BaconShopLockAll_Button` | `freeze` (a toggle: also unfreezes) |
| `TB_BaconShopTechUp0N_Button` | `level` |
| our hero power (raw CARDTYPE `HERO_POWER`) | `hero_power`, one per target (or one untargeted) |
| `BG36_Button_DarkGift` | `play` with `card_id` = the button, source `none` |
| card in hand | `play`, per target (or none) x per position (minions with board < 7) x per Choose One variant (SubOptions, read from the XML) |
| board minion, first untargeted entry | `reposition` to every other index (drag handle) |
| board minion, any further entry | `play` from the board (Activate), per target |
| shop card's own entry | skipped: it is the drag handle; buys come from the DragBuy helpers |

The helpers list every drop spot as a target (Bob, our hero, board and hand cards);
those are skipped and counted in `summary.options_skipped` (`buy_drop_target_*`,
`sell_drop_target_*`). A legal option whose entity or target is missing from the
snapshot is skipped and counted (`entity_not_in_state:*`, `*_target_not_in_state:*`).
Forced single-option decision points stay in.

## Inference

The states carry no choice. Between one decision point's `Options` block and the
next, the XML holds at most one top-level player action block (true for every pilot
decision point):

| method | action | confidence |
|---|---|---|
| `play_block` | `PLAY` block: entity + target (+ `subOption`) matched to the option | high |
| `play_block+zone_position` | `PLAY` of a hand minion; position = the minion's first `ZONE_POSITION` after it enters PLAY, inside the block | high |
| `move_minion_block` | `MOVE_MINION` block; position = the minion's last `ZONE_POSITION` inside the block | high |
| `no_action_turn_advanced` | no action and the next state's `raw_turn` is higher: `end_turn` (a timer expiry looks the same) | high |
| `no_action_last_decision_point` | no action after the game's last decision point: `end_turn` | medium |

Positions come from the XML, so the transition check against State Builder's next
state is independent evidence.

## Quarantine

A decision point that cannot be labeled is **excluded from the rows** and written to
`quarantine/<reviewId>.jsonl` with `reason` and `detail`. Nothing is guessed. Reasons:

- `noop_move`: a `MOVE_MINION` that did not change the position (dragged and dropped in place).
- `move_minion_in_shop_noop` / `_reorder`: a shop card was dragged and not bought.
- `buy_target_not_in_state_shop`, `sell_target_not_in_state_board`, `action_entity_not_in_state`,
  `action_entity_not_a_legal_option`, `play_from_shop`: the action does not map onto the state.
- `minion_play_position_unknown`, `suboption_unresolved`: missing position / Choose One data.
- `no_action_same_turn`, `multiple_actions`: nothing (or more than one thing) happened.
- `chosen_not_in_legal_options`: the inferred action is not in the enumerated list.
- `transition_failed:<type>`: the action is inconsistent with the next state.
- `dp_count_mismatch`: XML decision points != state rows (whole game quarantined).

Games with `placement != 1`, or with no states, replay or MMR, are dropped and listed
in `summary.games_dropped`.

## Choices (sidecar)

Hero, trinket, quest and discover picks are `Choices` blocks, not `Options` blocks, so
states.v1 has no decision point (and no snapshot) for them. They are written to
`choices/<reviewId>.jsonl`: `{game_id, after_dp_index, choice_kind, source_card,
offered: [{entity, card_id}], chosen, inference_method, confidence}`.
`choice_kind` is `hero` (Choices type 1 or all offered cards are heroes), `trinket`
(offered CARDTYPE `BATTLEGROUND_TRINKET`), `quest` (`BATTLEGROUND_QUEST_REWARD`),
else `discover`. The pick is `chosen_entities` (high) when `ChosenEntities` names one
offered card, `moved_to_hand` (medium) when there is no ChosenEntities and exactly one
offered card moved to HAND, else `unresolved`. Discover rows need a State Builder
decision point per `Choices` block (states.v2).

## Weight

`weight = 0.2 + 0.8 * p`, where `p` is the mid-rank percentile of the game's MMR among
all games with an MMR in the corpus manifest: `p = (#below + 0.5 * #equal) / N`. It rises
monotonically with MMR and stays inside (0.2, 1.0). There is no placement factor
(every row is from a 1st-place game). `summary.weight.corpus_examples` lists the weight
at the corpus min / p25 / median / p75 / max MMR.

## Checks (in `summary.json`)

1. **Label-match rate**: rows / decision points, per game and overall
   (`match_rate`, plus `match_rate_high_only`).
2. **Chosen in legal list**: enforced for every row (`label_game` raises otherwise);
   inferred actions not in the list are quarantined as `chosen_not_in_legal_options`.
3. **Transition check** (`transition_check`, pass / fail / na by action type), against
   the next state: buy (card left the shop, is in hand/board or became a golden), sell
   (left the board), reroll (shop entity set changed), freeze (`shop_frozen` toggled),
   level (tier + 1), reposition (minion at the final index, same board set), play from
   hand (left the hand; a minion sits at `position` and the other minions keep their
   order), hero power / Activate / Dark Gift (the state changed), end turn (`raw_turn`
   rose). When the turn ended before the next decision point, entity ids are re-issued,
   so card ids are compared instead (na where that is impossible). The last decision
   point is `na`. Failures are quarantined.
4. **Summary**: rows, rows per action type, match rate, confidence and method counts,
   quarantine count, reasons and examples, weight distribution (row min / median / max,
   per game), options per row, skipped option parts, choices sidecar counts.

Tests: `tests/test_replay_labels.py` (synthetic fixture
`tests/fixtures/synthetic_labels_replay.xml`). Set `HSBG_REPLAY_XML=<replay.xml.gz>` (and
optionally `HSBG_REPLAY_MANIFEST`) to also build states for a real replay and label it.
