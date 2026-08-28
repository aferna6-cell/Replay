"""Seed policy — where training seeds live, and the interval reserved for
Replay Benchmark v1 evaluation.

Every scheme below seeds ``hsbg_coach.bg_env.BGEnv`` — the same environment
the benchmark plays — so overlap would let a model train on the exact lobbies
it is later benchmarked on. Training code calls these helpers (rather than
inlining the arithmetic) so the separation tests in
``tests/test_benchmark.py`` exercise the real seed generation, not a copy.

KNOWN TRAINING SEED SCHEMES (BGEnv-seeded)
==========================================
additive (seed = base + offset, defaults keep everything under ~100k):
  * ``ml/bc.py`` demonstrations     base + i             (base 0, ~150 lobbies)
  * ``ml/bc.py`` DAgger rounds      base + 10000*r + i   (r=1.., ~80 lobbies)
  * ``ml/rl_common.evaluate_policy``  base + i           (base 9000, legacy eval)
  * ``hsbg_coach.bg_env.pace_curves`` base + i           (base 0)
multiplicative (blocks tile the whole seed space, one block per base seed):
  * ``ml/train_ppo.py`` episodes    base * 1000003 + k   (k = 1..iters*episodes)
  * ``ml/midgame_dataset.py``       base * 100003 + i    (i = 0..lobbies-1)

THE RESERVED EVALUATION INTERVAL
================================
[EVAL_SEED_START, EVAL_SEED_END] = [10_250_000, 10_299_999]  (50_000 seeds)

Why these schemes cannot reach it under any current default or reasonable
configuration:
  * additive schemes: with their shipped default bases they top out around
    10^5, four orders of magnitude below the interval.
  * PPO: no multiple of 1000003 lies inside the interval; the nearest block
    start below it is 1000003*10 = 10_000_030, which is 249_970 seeds away.
    So for EVERY base seed, a single PPO run would need 249_970 or more
    episodes (defaults: 640) before any episode seed could land inside.
  * midgame dataset: no multiple of 100003 lies inside; the nearest block
    start below is 100003*102 = 10_200_306, 49_694 seeds away. So for EVERY
    base seed, one generation run would need more than 49_694 lobbies
    (defaults: 300; calibrate: 60) to reach the interval.

This is separation under stated bounds, NOT a mathematical guarantee: a PPO
run with base seed 10 and >=249_970 episodes, a midgame run with base seed 102
and >49_694 lobbies, or an additive base chosen near 10.25M WOULD collide.
``check_training_range()`` makes those cases loud at training time, and the
benchmark refuses to run outside the reserved interval.

WHAT FUTURE TRAINING CODE MUST PRESERVE
=======================================
Any new code that seeds BGEnv for training/data generation must (a) derive
its seeds through a helper in this module, (b) call ``check_training_range``
on the full planned seed span, and (c) never intentionally use seeds inside
the reserved interval. Non-BGEnv seeding (``ml/econ_env.py``, the combat-sim
datasets in ``ml/data.py``) plays a different environment and cannot leak
benchmark lobbies; it is out of scope here.
"""

import sys

EVAL_SEED_START = 10_250_000
EVAL_SEED_END = 10_299_999                 # inclusive; 50_000 evaluation seeds

_PPO_STRIDE = 1_000_003
_MIDGAME_STRIDE = 100_003
_DAGGER_STRIDE = 10_000


# --- benchmark side -----------------------------------------------------------
def eval_game_seed(base: int, i: int) -> int:
    """Seed for benchmark game i (deterministic: base + i)."""
    return base + i


def validate_eval_range(base: int, games: int) -> None:
    """Reject a benchmark request that leaves the reserved interval."""
    if games < 1:
        raise ValueError(f"--games must be >= 1, got {games}")
    last = eval_game_seed(base, games - 1)
    if base < EVAL_SEED_START or last > EVAL_SEED_END:
        raise ValueError(
            f"requested evaluation seeds {base}-{last} exceed the reserved "
            f"Replay Benchmark v1 interval [{EVAL_SEED_START}, "
            f"{EVAL_SEED_END}] — pick a base seed and game count that fit "
            f"(capacity {EVAL_SEED_END - EVAL_SEED_START + 1} games)")


def overlaps_eval_range(lo: int, hi: int) -> bool:
    """Does the inclusive seed span [lo, hi] intersect the reserved interval?"""
    return lo <= EVAL_SEED_END and hi >= EVAL_SEED_START


def check_training_range(scheme: str, lo: int, hi: int) -> bool:
    """Loud warning when a planned training seed span would touch the
    reserved evaluation interval. Returns True when it overlaps. A warning,
    not an error, so existing training workflows keep running — but results
    trained on benchmark seeds must not be benchmarked."""
    if overlaps_eval_range(lo, hi):
        print(f"WARNING [{scheme}]: planned training seeds {lo}-{hi} overlap "
              f"the reserved Replay Benchmark v1 evaluation interval "
              f"[{EVAL_SEED_START}, {EVAL_SEED_END}]. A model trained on "
              f"these lobbies must NOT be scored with ml.benchmark — choose "
              f"a different base seed.", file=sys.stderr)
        return True
    return False


# --- training-side seed derivation (the real schemes, called by trainers) -----
def bc_lobby_seed(base: int, i: int) -> int:
    """ml/bc.py demonstration lobby i."""
    return base + i


def dagger_round_base(base: int, round_index: int) -> int:
    """ml/bc.py DAgger round r (1-based) lobby base; lobby i then uses
    bc_lobby_seed(dagger_round_base(base, r), i)."""
    return base + round_index * _DAGGER_STRIDE


def ppo_episode_seed(base: int, episode_index: int) -> int:
    """ml/train_ppo.py rollout seed for the k-th episode (1-based)."""
    return base * _PPO_STRIDE + episode_index


def midgame_lobby_seed(base: int, i: int) -> int:
    """ml/midgame_dataset.py lobby i (0-based)."""
    return base * _MIDGAME_STRIDE + i


def legacy_eval_seed(base: int, i: int) -> int:
    """ml/rl_common.evaluate_policy episode i (training-loop progress eval)."""
    return base + i
