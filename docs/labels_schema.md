# `labels.v2`: decision-point training rows

`hsbg_coach/replay_labels.py` turns State Builder's `states.v2` rows
(`docs/replay_states_v1.md`) into weighted behaviour-cloning rows:
`(state, every legal option, chosen option)`, for shopping decisions (`kind="options"`)
and for picks (`kind="choice"`: hero, discover, Dark Discovery, trinket, quest, other).

```
python -m hsbg_coach.replay_labels --states data/firestone/states/v2 \
    --replays data/firestone/raw --manifest data/firestone/raw/manifest.json \
    --out data/firestone/labels/v2 --workers 6
```

Everything under `data/firestone/` is gitignored and must never be committed.

Inputs: the states folder (`<reviewId>.jsonl.gz`, plus State Builder's
`quarantine/<reviewId>.json`), the replay folder (`<reviewId>.xml.gz`, read again for
the player's actions and picks), the manifest of games to label (placement, MMR,
`creationDate`) and the corpus manifest (MMR percentiles for weights; defaults to
`--manifest`). `--workers N` labels games in N processes.

`--current-build N` is the build treated as the current patch. When it is omitted,
that build is the highest `buildNumber` among games in the corpus manifest
(`--corpus-manifest`, otherwise `--manifest`). The value actually used is written
to `summary.json` as `current_build` and under `inputs.current_build`.

Outputs:

- `<out>/<reviewId>.jsonl.gz`: labeled rows (one per decision point that could be labeled).
- `<out>/quarantine/<reviewId>.jsonl`: decision points that could not be labeled, with a reason.
- `<out>/summary.json`: every check and count listed under "Checks".

The v1 `choices/` sidecar is gone: picks are rows now.

## Row

One row per State Builder decision point (`dp_index`, chronological over options and
choice rows), keyed by `(game_id, dp_index)`.

| field | type | notes |
|---|---|---|
| `game_id`, `build`, `mmr` | str, int, int | copied from the state row |
| `current_patch` | bool | true when the row's `build` equals the current build: `--current-build`, or, if that flag is omitted, the highest `buildNumber` in the corpus manifest (`--corpus-manifest`, else `--manifest`) |
| `turn`, `dp_index` | int | copied from the state row |
| `placement` | int | manifest placement; always 1 (other games are dropped) |
| `created_at` | str | manifest `creationDate` as UTC ISO-8601 with milliseconds, e.g. `2026-09-27T20:59:53.982Z` |
| `created_ts` | int | the same instant in epoch milliseconds (from `creationDate`; `creationTimestamp` only when there is no `creationDate`) |
| `state` | object | the state row's `snapshot` (`Snapshot.to_dict()`), unchanged |
| `options` | [option] | the legal options at this decision point, expanded (see below) |
| `chosen` | int | index into `options`; always valid (the run raises otherwise) |
| `inference_method` | str | see "Inference" |
| `confidence` | `high` / `medium` / `low` | see "Inference"; no `low` rows are emitted |
| `weight` | float | MMR weight, see "Weight" |
| `flags` | [str] | additive, usually empty; see "Flags" |

Row-level state fields that are not in the snapshot (`kind`, `options_index`,
`lobby_tribes`, row-level `hero_power`, `dark_discovery`, `raw_turn`, `choice`) are not
copied; join the states file on `(game_id, dp_index)` if needed. A choice row is the
row whose chosen option has `type == "discover"`.

### Option

```json
{"type": "buy|sell|reroll|freeze|level|hero_power|play|discover|end_turn|reposition",
 "card_id": "str|null",
 "source": {"zone": "board|shop|hand|choice|none", "slot": "int|null"},
 "target": {"zone": "board|shop|hand|none", "slot": "int|null"},
 "position": "int|null",
 "src_entity": "int|null", "target_entity": "int|null",
 "choice_kind": "hero|discover|dark_discovery|trinket|quest|other|null"}
```

- **Slots** are 0-based indexes into the snapshot lists (`board`, `hand`, `shop`).
  State Builder sorts those lists by `ZONE_POSITION`. `hand` holds every hand card,
  spells included. **Shop spells** use slot `len(shop) + j` (j = index into
  `shop_spells`, which is sorted by shop position).
- **`position`**: for `play` of a hand minion, the insertion index `0..len(board)`
  (= ZONE_POSITION - 1 after the play). A MAGNETIC minion dropped at `position` p to the
  left of a Mech magnetizes onto `board[p]` (same action in the UI). For `reposition`,
  the final index (never equal to the source slot). Null otherwise.
- **`src_entity` / `target_entity`**: raw XML entity ids, valid within this row's
  snapshot only (debug). For buy/sell, `src_entity` is the card bought/sold (the
  DragBuy/DragSell helper id is dropped). For discover, the offered card entity.
- **`card_id`**: the card acted on (the shop card for buy, the board minion for sell,
  the Choose One variant for a Choose One play, the hero power for hero_power, the
  offered card for discover). Null for `reroll`, `freeze`, `level` and `end_turn`.
- **`choice_kind`**: State Builder's `choice.choice_kind` on `discover` options, else null.
- Costs are not options fields: read `state.shop[].buy_cost`, `state.shop_spells[].buy_cost`,
  `state.reroll_cost`, `state.level_cost` and `state.hero_power`.

### Option enumeration: options rows

Only server-legal options (`error=-1`, as kept by State Builder) are used. Each is
expanded into one labeler option per distinct target, board position and Choose One
variant, then de-duplicated on `(type, card_id, source, target, position)` (plus
`target_entity` when the target is outside board/shop/hand, e.g. our hero).

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
| card in hand | `play`, per target (or none) x per position (minions with board < 7) x per Choose One variant (`options[].sub_options`; the XML SubOptions for states.v1 rows) |
| board minion, first untargeted entry | `reposition` to every other index (drag handle) |
| board minion, any further entry | `play` from the board (Activate), per target |
| shop card's own entry | skipped: it is the drag handle; buys come from the DragBuy helpers |

The helpers list every drop spot as a target (Bob, our hero, board and hand cards);
those are skipped and counted in `summary.options_skipped` (`buy_drop_target_*`,
`sell_drop_target_*`). A legal option whose entity or target is missing from the
snapshot is skipped and counted (`entity_not_in_state:*`, `*_target_not_in_state:*`).
Forced single-option decision points stay in.

### Option enumeration: choice rows

One `discover` option per `choice.cards` entry, in offer order:
`{"type": "discover", "card_id": card_id, "source": {"zone": "choice", "slot": i},
"src_entity": entity_id, "choice_kind": choice_kind}`. Every State Builder choice row
has `min = max = 1`.

## Inference

The states carry no choice.

**Options rows.** Between one `Options` block and the next, the XML holds at most one
top-level player action block:

| method | action | confidence |
|---|---|---|
| `play_block` | `PLAY` block: entity + target (+ `subOption`) matched to the option | high |
| `play_block+zone_position` | `PLAY` of a hand minion; position = the minion's first `ZONE_POSITION` after it enters PLAY, inside the block | high |
| `move_minion_block` | `MOVE_MINION` block; position = the minion's last `ZONE_POSITION` inside the block | high |
| `no_action_turn_advanced` | no action and the next options row's `raw_turn` is higher: `end_turn` (a timer expiry looks the same) | high |
| `no_action_last_decision_point` | no action after the game's last options row: `end_turn` | medium |

Options rows are paired with the XML `Options` blocks in order (`options_index`; a
re-sent `Options` id counts once, as in State Builder).

**Choice rows.** Each choice row is paired, in order, with the next XML `Choices` block
offering exactly the same entity ids, and the pick is the offered entity named by the
`ChosenEntities` that answers it:

| method | pick | confidence |
|---|---|---|
| `chosen_entities` | the one offered entity in `ChosenEntities` | high |
| `moved_to_hand` | no `ChosenEntities`, and exactly one offered card moved to HAND before the next `Options` | medium |

## Quarantine

A decision point that cannot be labeled is **excluded from the rows** and written to
`quarantine/<reviewId>.jsonl` with `kind`, `reason` and `detail`. Nothing is guessed.
Reasons:

- Not decisions (left out of `decision_match_rate`):
  - `noop_move`: a `MOVE_MINION` that did not change the position (dragged and dropped in place).
  - `move_minion_in_shop_noop` / `_reorder` (and `_hand_`): a shop or hand card was
    dragged and not played or bought.
  - `choice_offer_resent`: the same offer was re-sent unchanged (after a re-dump) before
    any pick. It is one decision: the pick is labeled on the re-sent row.
  - `no_action_same_turn`: the `Options` block was followed by the next one in the same
    turn with no action block at all (the server re-sent the options under a new id).
- `buy_target_not_in_state_shop`, `sell_target_not_in_state_board`, `action_entity_not_in_state`,
  `action_entity_not_a_legal_option`, `play_from_shop`: the action does not map onto the state.
- `minion_play_position_unknown`, `suboption_unresolved`: missing position / Choose One data.
- `multiple_actions`: more than one action block before the next `Options`.
- `choice_not_in_replay`, `choice_pick_unresolved`, `choice_multiple_picked`,
  `chosen_entities_not_offered`: the pick cannot be read from the XML.
- `chosen_not_in_legal_options`: the inferred action is not in the enumerated list.
- `transition_failed:<type>`: the action is inconsistent with the next options row.
- `dp_count_mismatch`: XML `Options` blocks != options rows (whole game quarantined).

Games with `placement != 1`, a State Builder quarantine file, or no states, replay or
MMR are dropped and listed in `summary.games_dropped`.

## Flags

`golden_dark_gift_first_only`: the snapshot holds a golden tripled from 2+ Dark-Gifted
copies. It carries every copy's gift, but State Builder's `dark_gift` names only the
first one found. Detected heuristically (a golden `X_G` appears while the previous row
held 2+ gifted `X` and fewer remain), then tracked by card id while a gifted `X_G` is
held. The row is still labeled.

## Weight

`weight = 0.2 + 0.8 * p`, where `p` is the mid-rank percentile of the game's MMR among
all games with an MMR in the corpus manifest: `p = (#below + 0.5 * #equal) / N`. It rises
monotonically with MMR and stays inside (0.2, 1.0). There is no placement factor
(every row is from a 1st-place game). `summary.weight.corpus_examples` lists the weight
at the corpus min / p25 / median / p75 / max MMR.

## Checks (in `summary.json`)

1. **Label-match rate**: rows / decision points, per game and overall (`match_rate`,
   `match_rate_high_only`), and `decision_match_rate` = rows / (decision points - not
   decisions). `worst_games` / `worst_games_decision_match` list the 10 lowest games.
2. **Chosen in legal list**: enforced for every row (`label_game` raises otherwise);
   inferred actions not in the list are quarantined as `chosen_not_in_legal_options`.
3. **Transition check** (`transition_check`, by action type; choice rows as
   `discover:<choice_kind>`).
   - Options rows (pass / fail / na), against the next options row: buy (card left the
     shop, is in hand/board or became a golden), sell (left the board), reroll (shop
     entity set changed), freeze (some shop minion's `FROZEN` changed; na when the shop
     has only spells, which carry no frozen flag), level (tier + 1), reposition
     (minion at the final index, same board set), play from hand (left the hand; a
     minion sits at `position` counted among the minions that were already on the board
     and are still there, which keep their order (a battlecry may destroy one, a summon
     may add tokens), or a MAGNETIC minion is gone and `board[position]` is still
     there), hero power / Activate / Dark Gift (the state changed, card tags and costs
     included), end turn (`raw_turn` rose). When the turn ended
     before the next options row, entity ids are re-issued, so card ids are compared
     instead (na where that is impossible). The last options row is `na`. Failures
     are quarantined.
   - Choice rows (pass / unverified / na): the picked entity is in hand/board at the
     next options row, or its card id (or a golden / upgraded `card_id*`) is the hero,
     a hand/board card, a trinket, a quest or the hero power. The pick itself is the
     server's `ChosenEntities`, so a pick with no visible trace (spent on pick,
     tripled, transformed) is `unverified`, not quarantined.
4. **Summary**: rows (options / choice), rows per action type and per choice kind,
   confidence and method counts, quarantine count, reasons (by row kind) and examples,
   weight distribution, `created_at_range`, `current_build`, builds, flags, options per row, skipped
   option parts.

Tests: `tests/test_replay_labels.py` (synthetic fixture
`tests/fixtures/synthetic_labels_replay.xml`). Set `HSBG_REPLAY_XML=<replay.xml.gz>` (and
optionally `HSBG_REPLAY_MANIFEST`) to also build states for a real replay and label it.