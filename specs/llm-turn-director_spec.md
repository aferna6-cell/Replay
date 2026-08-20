# LLM Turn Director — spec

Status: agreed 2026-08-20 (owner interview, this session). Phase 1 (Tier7
bridge) implemented; phases 2-5 pending.

## Goal

An open-source LLM drives move-by-move coaching in live Battlegrounds games:
one suggested move at a time with a one-line why, backed by a whole-turn and
whole-game plan it maintains silently, grounded in **current-patch HDT/Tier7
data**, and improving from the owner's own games via automatic between-game
review.

## Hard requirements (owner's words, distilled)

1. **One move at a time** in the overlay: `MOVE — one-line why`. The model
   plans the turn and game ahead, but surfaces only the next move.
2. **Never float gold.** Suggesting "end turn" with spendable gold is a bug
   unless explicitly justified (infinite economy, deliberate freeze/save).
   Enforced by a deterministic validator between the LLM and the overlay —
   not trusted to the model.
3. **Plans and lines come from HDT data.** Tier7 (lobby-conditioned hero
   stats, first-place comps + key minions for this lobby's tribes, trinkets,
   dark gifts) is the primary source; Firestone aggregates are the fallback.
4. **Current patch ONLY.** Meta pack embeds the patch ID; stale-across-patch
   data is refused, not silently served. Firestone pulls pin
   `--period last-patch`; Tier7 is current by construction.
5. **Flexible tribes.** The game plan ranks the lobby's tribes and names
   enablers ("Elementals S-tier here: find Brann + …"), but commits late and
   plays what the shops offer. Pivot triggers are part of the plan and the
   Director re-evaluates them every turn.
6. **Fast.** Move + why in ~2s; slower is acceptable for rarer, heavier
   decisions (trinket/hero picks). Engine math is instant; only the LLM call
   costs time.
7. **Between-game review.** When a game ends, a background grader replays the
   trajectory vs the known placement, grades the Director's own suggestions
   ("that roll was bad — buying X was better"), and appends lessons the
   Director reads next game. Runs inside the queue/hero-pick window.
8. **Rules are given, not remembered.** Game rules (tavern costs, triples,
   pool sizes, keywords) + the committed card KB go into the model's context
   from files; model memory is never trusted for rules or the meta.

## Non-goals

- Combat-sim accuracy work (owner: "not interested in combat simulation").
  The sim stays as-is; investment goes to the decision layer.
- LLM end-to-end play. The engine computes candidates; the LLM directs.
- Mid-game weight updates. Learning = lessons file (immediately) + periodic
  LoRA retrain (weekly-ish, off-device).

## Architecture

```
Power.log ─► parser/bg state ─► L1 engine (ms): legal actions, gold plan,
                                  eval-net deltas, beam-searched full-gold lines
                                        │ candidates + state
                                        ▼
   meta pack (Tier7-first) ──► L2 Turn Director (LLM, ~1-2s):
   game plan + lessons file       pick ONE move + one-line why,
                                  maintain turn/game plan, pivot checks
                                        │ suggestion
                                        ▼
                              validator: no-float-gold, legality,
                              card names ∈ KB  ─► overlay panel
game end ─► L4 reviewer (background): grade suggestions vs outcome,
            append lessons + training examples ─► next game / LoRA corpus
```

- **L3 Game Plan** (lobby start, ~5s budget): lobby tribes + Tier7 hero pick
  stats → strategy memo (plan A/B, enablers to hunt, pivot triggers). Stored
  as state; Director may revise it and must say so.
- **Serving**: model-server-agnostic client — local (Ollama/llama.cpp, small
  Qwen-class model) or hosted open-weights (Groq/Together/OpenRouter).
  `llm-bench` CLI measures the owner's ThinkPad (with HS running) and the
  numbers pick the deployment. LoRA adapters work on either path.
- **Training** never runs on the ThinkPad: rented GPU or a fine-tune API,
  fed by the reviewer's corpus + tier7_log.jsonl.

## Data sources (priority order)

1. Tier7 via `hsbg_coach/tier7.py` (this repo, phase 1) — lobby-conditioned,
   current patch, owner's own subscription; every response logged to
   `data/tier7_log.jsonl` as ML context. Dark-gift endpoint: check HDT source
   the same way the six known endpoints were recovered; else manual seed.
2. Firestone `refresh-stats --period last-patch` — fallback + pace curves.
3. Owner's recorded games (`data/*.jsonl`) — the personalization signal.

## Phases

1. **Tier7 bridge** — module + CLI + tests + ADR (done, this branch); live
   calibration on the owner's PC closes the `# CALIBRATE` list; dark-gift
   endpoint check.
2. **Meta pack + Game Plan** — compact current-patch context builder
   (tribes/heroes/comps/enablers/trinkets/dark gifts, Tier7-first) + the
   lobby-start memo.
3. **Turn Director** — LLM move+why over engine candidates; server-agnostic
   client; `llm-bench`; the no-float-gold validator.
4. **Overlay wiring** — `watch` drives the Director; panel = move + why +
   current plan line.
5. **Review loop** — between-game grader, lessons file, corpus builder,
   weekly retrain script (LoRA off-device).

## Open items

- ThinkPad local-vs-hosted: decided by `llm-bench` results, not assumption.
- Dark-gift endpoint existence (phase 1 calibration).
- Tier7 hero-pick request field names (`# CALIBRATE` in tier7.py) — first
  live 400 names any mismatch.
