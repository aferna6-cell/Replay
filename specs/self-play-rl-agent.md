# Self-Play RL Battlegrounds Agent — Design & Feasibility

Status: **Phase 0 built + validated; Phase 1 running.**
- Phase 0 env: `hsbg_coach/bg_env.py` — validated against Firestone pace /
  scaling / alive-by-turn aggregates (the §8 gate passed; see README).
- State encoder: per-minion tokens + set-transformer (`ml/tokens.py`,
  `ml/set_net.py`) — the §6b architecture, trained on env mid-game states
  (`ml/midgame_dataset.py`, `ml/train_set_net.py`, calibration in
  `ml/calibrate.py`).
- Phase 1: policy/value net with masked 28-action head (`ml/policy_net.py`),
  BC warm start from the greedy baseline (`ml/bc.py`), PPO + league
  (`ml/train_ppo.py`). Gate: beat random + greedy fields.
Remaining phases (wider cards, heroes/trinkets, scale) below are future work.

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

## 6b. Deep-learning architecture

This whole agent **is** deep learning — neural nets for the policy and value
functions (classical ML = the shallow value/synergy models we considered for
aggregate data). BG's structure favors specific deep designs:

**Card embeddings (representation learning).** Learn a dense vector per minion
instead of hand-coded tier/tribe/keyword features. Two sources:
- *Now, no env:* **card2vec** — learn embeddings from **co-occurrence in the 20k
  winning boards** (cards that win together end up near each other), the same
  idea as word2vec on sentences. Doable today on data we have; also upgrades the
  current synergy layer.
- *In the RL net:* an embedding table trained end-to-end with the policy.
These let the model **generalize across the card pool** by learned properties —
directly attacking the "shallow card understanding" gap.

**Set/attention encoder (the natural fit).** Board and shop are *sets* of
minions. A **self-attention / Transformer encoder over the minions** is the
standout architecture here: attention literally models minion↔minion interactions
— i.e. **synergy emerges from the network** rather than being hand-written.
Permutation-invariant (DeepSets is the simpler fallback). A small Transformer over
[board ∪ shop ∪ context tokens] is the recommended encoder.

**Policy + value heads.** Shared encoder → (a) **autoregressive policy** (pick
action type, then target, with legal-action masking — mirrors `bg.ActionType`),
(b) **scalar value** head predicting expected placement.

**Search + net (AlphaZero-flavored), with a twist.** Our fast combat sim makes
lookahead cheap, so MCTS guided by the net is attractive. BUT BG is **imperfect
information + RNG** (hidden enemy boards, random shops), so plain AlphaZero MCTS
doesn't apply — use **Information-Set MCTS (ISMCTS)** or determinized rollouts.

**Sequence-model / offline routes (future, data-gated).** **Decision Transformer**
and offline-RL (CQL) frame control as sequence modeling over trajectories — elegant,
but they **need trajectory datasets we don't have at scale**. Candidate later, if
we accumulate enough `Power.log` games or self-play data.

**Honest caveat:** deep learning is the *function approximator*, **not a data
source**. It does not remove the Phase 0 requirement — deep RL still needs the
self-play environment; deep offline methods still need trajectories. Picking
"deep learning" changes the *model class*, not the data constraint.

## 7. How existing assets plug in

| Asset | Role in the agent |
|---|---|
| `sim.py` / Firestone bridge | Combat dynamics in the env + lookahead value |
| `ml/` combat value net | **Built.** Learned fast approximation of combat (win/tie/loss) for the RL value/lookahead; trained on unlimited sim-generated data (win%-MAE ~0.05). First DL component; proves the deep approach works. |
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

**One deep-learning artifact is buildable now, independent of Phase 0:**
**card2vec embeddings** from the 20k winning boards (§6b). It needs no
environment and no trajectories — just the data we have — and it pays off twice:
it upgrades today's synergy layer (learned card relationships > regex) and
becomes the input embedding for the future RL net. Low-risk, high-reuse; a good
parallel track to Phase 0.

## Cross-refs
- `specs/hsbg-coach_spec.md` — data layer + why full games aren't available
- `hsbg_coach/sim.py`, `effects.py` — combat engine (env dynamics)
- `hsbg_coach/cards.py`, `final_boards.py`, `pace.py` — features + reward priors
- `hsbg_coach/recorder.py` — trajectory schema (optional warm-start)
