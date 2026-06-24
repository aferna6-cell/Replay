# Self-Play RL Battlegrounds Agent — Design & Feasibility

Status: **Design / feasibility (not started).** This scopes the "learn to play"
model. Read the honest-assessment section before committing engineering time.

## 0. Why this approach

A decision policy needs full game records — **(state → action → outcome)
sequences**. Population data (Firestone/HSReplay) does not provide them (only
aggregates + end-of-game board snapshots; see `specs/hsbg-coach_spec.md` §5).
The only way to get full games without millions of human logs is to have the
agent **generate its own** by playing in a simulator: **self-play reinforcement
learning**. Everything we've built becomes the environment's combat engine,
state features, and reward priors.

This is the AlphaStar/AlphaZero-shaped problem flagged at project start. It is
**research-grade**: a proof-of-concept that plays *simplified* BG well is
achievable; a superhuman *full* BG agent is a major, uncertain, compute-heavy
project. This doc is written so we can decide with eyes open.

## 1. Problem framing

Battlegrounds as an MDP (really a POMDP — partially observable):

- **Episode** = one game (8 players, ~10-15 turns), from the agent's seat.
- **Two interleaved phases per turn:** Recruit (economy/shop decisions) →
  Combat (auto-resolved). The agent acts in Recruit; Combat is environment
  dynamics.
- **Partial observability:** you don't see opponents' current boards — only the
  board you last fought, and the shared lobby info (tribes, others' health/tier).
- **Objective:** maximize final **placement** (1st…8th). Long horizon, sparse
  terminal reward.

## 2. The environment (the make-or-break component)

A faithful-enough BG simulator. **This is ~70% of the work and the #1 risk** —
if the env diverges from real BG, the agent learns to exploit the sim, not play
BG (the sim2real gap).

| Subsystem | Plan | Have it? |
|---|---|---|
| **Combat resolution** | Reuse `sim.py` (or the Firestone bridge) | ✅ built |
| **Economy** | Gold (3→10), tavern costs+discount, buy/sell/roll/freeze — rules in `economy.py`/wiki | rules known |
| **Shop generation** | Draw from the minion pool by tavern tier, restricted to the lobby's tribes; respect **pool copy counts** (BG has finite copies per minion) | needs pool-size data |
| **Card pool + mechanics** | Card features from `cards.py`; mechanics in combat from `effects.py`/Firestone | partial (long tail) |
| **Opponents** | Self-play **league** (pool of past agent versions) + scripted baselines | to build |
| **Lobby structure** | 8 seats, random tribe subset, pairing each combat | to build |
| **Heroes / hero powers / trinkets / quests / anomalies** | **Defer to later phases** — start with a vanilla hero, no trinkets/anomalies | defer |

**Fidelity strategy:** start deliberately **simplified** (core economy + combat +
a curated card subset whose effects are fully modeled), make RL tractable, then
widen coverage. Validate the env against Firestone aggregates (do random/greedy
agents produce the same rough pace/board-stat curves we measured?).

## 3. State representation (observation)

Encode the visible state as fixed-size tensors:

- **Your board:** up to 7 minions × features (tier, tribe one-hot, attack,
  health, keyword flags) — from `cards.py` knowledge.
- **Shop:** offered minions × same features.
- **Hand**, **gold**, **tavern tier**, **health**, **turn**.
- **Lobby:** tribes available, opponents' health/tier, last-seen enemy boards.

Sets (board/shop) → permutation-invariant encoder (DeepSets / small transformer).

## 4. Action space (structured — needs masking)

Recruit actions: `buy(slot)`, `sell(slot)`, `play(hand_slot)`, `reposition(perm)`,
`roll`, `freeze`, `tier_up`, `hero_power`, `end_turn`. Combat positioning folds
into end-of-recruit ordering.

- **Large, variable, structured** → use **action masking** (only legal actions)
  and an **autoregressive / hierarchical** policy head (pick action type, then
  target). This mirrors `bg.ActionType` (already enumerated).

## 5. Reward

- **Terminal:** placement-based, e.g. `+(4.5 − placement)` so 1st = +3.5, 8th =
  −3.5 (zero-mean). This is the true objective.
- **Shaping (use our data, carefully):** small dense signals to fight sparsity —
  on-pace leveling (`pace.py`), combat win-prob (`sim.py`), comp-fit /
  winning-board similarity (`final_boards.py`), surviving with health. **Risk:**
  bad shaping teaches greedy/suboptimal play; keep shaping weights small and
  anneal toward pure placement.

## 6. Algorithm

Two viable routes; recommend a hybrid:

- **PPO** (model-free, standard, robust) as the backbone.
- **Lookahead via the combat sim:** because we can *simulate combat cheaply*, the
  value estimate and shop decisions can use short-horizon search (evaluate
  "buy X vs level" by simulating the resulting board's win-prob). This is an
  AlphaZero-flavored advantage most RL problems don't have — exploit it.
- **Self-play league** (AlphaStar lesson): train against a *pool* of past agents,
  not just the latest, to avoid strategy collapse / rock-paper-scissors cycling.

## 7. How existing assets plug in

| Asset | Role in the agent |
|---|---|
| `sim.py` / Firestone bridge | Combat dynamics in the env + lookahead value |
| `cards.py` | State features (tier/tribe/keywords) |
| `final_boards.py` | Reward shaping (similarity to known winning boards); curriculum targets; sanity check |
| `pace.py` | Reward shaping (on-curve leveling); env validation |
| `stats.py` comps/heroes | Curriculum + evaluation ("does it build known S-tier comps?") |
| `recorder.py` schema | Optional **imitation warm-start** from human logs later |

## 8. Phased plan (each phase is a go/no-go gate)

- **Phase 0 — Environment scaffold (the feasibility gate).** Economy + shop
  generation + integrate combat sim; Gym-style API; random + greedy baselines;
  validate pace/board curves vs Firestone. *If we can't build a faithful-enough
  env, stop here.* Small, decisive, reuses the combat sim.
- **Phase 1 — Simplified RL.** No heroes/trinkets/anomalies; curated card subset;
  PPO; reward = placement + light shaping. **Goal: beat random + greedy.**
- **Phase 2 — Self-play league + value net + lookahead; widen card set.**
- **Phase 3 — Add heroes / hero powers / trinkets; expand mechanics; scale
  compute.**
- **Phase 4 — Evaluate vs the heuristic engine and (optionally) the user's own
  play; consider human-log warm-start.**

## 9. Honest feasibility & risks

- **Environment fidelity (highest risk):** divergence → the agent games the sim.
  Mitigate with the simplified-first strategy + validation against real
  aggregates.
- **Compute:** real RL needs serious parallelism + GPU time. A simplified PoC is
  hobbyist-feasible; a strong full-BG agent is expensive (think many GPU-days+).
- **Action/state complexity:** large structured action space; masking +
  hierarchical policy are non-trivial.
- **Mechanics long tail:** hundreds of unique card effects — same coverage
  problem as the combat sim. Start simplified; lean on the Firestone bridge.
- **Meta drift:** every patch changes cards; a trained agent decays and needs
  retraining.
- **Reward shaping:** can entrench bad habits if mis-weighted.
- **Realistic outcome:** a PoC that plays simplified BG sensibly (beats greedy
  baselines, builds coherent comps) is a credible target. Superhuman full BG is a
  research program, not a sprint.

## 10. Recommendation / decision gate

Build **Phase 0 only**, as a pure feasibility test: it's small, it reuses the
combat sim we already have, and it answers the make-or-break question (can we
simulate BG faithfully enough?) before any RL or compute spend. Reassess at the
Phase 0 → Phase 1 gate with a working environment + baseline agents in hand.

If Phase 0's environment can't reproduce the real pace/board curves we measured
from Firestone, that's the signal to stop and stay with the simulation+search
heuristic engine instead.

## Cross-refs
- `specs/hsbg-coach_spec.md` — data layer + why full games aren't available
- `hsbg_coach/sim.py`, `effects.py` — combat engine (env dynamics)
- `hsbg_coach/cards.py`, `final_boards.py`, `pace.py` — features + reward priors
- `hsbg_coach/recorder.py` — trajectory schema (optional warm-start)
