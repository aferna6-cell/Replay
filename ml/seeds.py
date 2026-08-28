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

THE RESERVED INTERVALS — TRAIN / DEV / TEST SPLIT
=================================================
TEST  [EVAL_SEED_START, EVAL_SEED_END] = [10_250_000, 10_299_999] (50_000 seeds)
DEV   [DEV_SEED_START,  DEV_SEED_END]  = [10_550_000, 10_599_999] (50_000 seeds)

TRAIN (everything else, via the schemes above) learns parameters. DEV is for
model development: diagnosing checkpoints, comparing experimental variants,
choosing hyperparameters and training durations (``ml/dev_benchmark.py``).
TEST is Replay Benchmark v1 (``ml/benchmark.py``): the held-out final test
set, used sparingly for final confirmation only — never for iterating on
models. Repeatedly steering development by TEST results would silently turn
the test set into a dev set and invalidate the published baselines.

Why the training schemes cannot reach either interval under any current
default or reasonable configuration:
  * additive schemes: with their shipped default bases they top out around
    10^5, four orders of magnitude below both intervals.
  * PPO: no multiple of 1000003 lies inside either interval; the nearest
    block start below TEST is 1000003*10 = 10_000_030, 249_970 seeds away
    (and 549_970 below DEV). So for EVERY base seed, a single PPO run would
    need 249_970+ episodes (defaults: 640) before any episode seed could
    land in TEST, and 549_970+ to land in DEV.
  * midgame dataset: no multiple of 100003 lies inside either interval; the
    nearest block starts below are 100003*102 = 10_200_306 (49_694 below
    TEST) and 100003*105 = 10_500_315 (49_685 below DEV). So for EVERY base
    seed, one generation run would need >49_694 (TEST) / >49_685 (DEV)
    lobbies (defaults: 300; calibrate: 60) to reach them.

This is separation under stated bounds, NOT a mathematical guarantee: a PPO
run with base seed 10 and >=249_970 episodes, a midgame run with base seed 102
and >49_694 lobbies, or an additive base chosen near the intervals WOULD
collide. ``check_training_range()`` makes those cases loud at training time,
and both evaluation CLIs refuse to run outside their reserved interval.

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

EVAL_SEED_START = 10_250_000               # TEST — Replay Benchmark v1
EVAL_SEED_END = 10_299_999                 # inclusive; 50_000 seeds
DEV_SEED_START = 10_550_000                # DEV — model development eval
DEV_SEED_END = 10_599_999                  # inclusive; 50_000 seeds

_PPO_STRIDE = 1_000_003
_MIDGAME_STRIDE = 100_003
_DAGGER_STRIDE = 10_000


# --- evaluation side ----------------------------------------------------------
def eval_game_seed(base: int, i: int) -> int:
    """Seed for evaluation game i (deterministic: base + i)."""
    return base + i


def _validate_range(base: int, games: int, start: int, end: int,
                    label: str) -> None:
    if games < 1:
        raise ValueError(f"--games must be >= 1, got {games}")
    last = eval_game_seed(base, games - 1)
    if base < start or last > end:
        raise ValueError(
            f"requested evaluation seeds {base}-{last} exceed the reserved "
            f"{label} interval [{start}, {end}] — pick a base seed and game "
            f"count that fit (capacity {end - start + 1} games)")


def validate_eval_range(base: int, games: int) -> None:
    """Reject a TEST (Benchmark v1) request that leaves its interval."""
    _validate_range(base, games, EVAL_SEED_START, EVAL_SEED_END,
                    "Replay Benchmark v1 TEST")


def validate_dev_range(base: int, games: int) -> None:
    """Reject a DEV evaluation request that leaves the DEV interval."""
    _validate_range(base, games, DEV_SEED_START, DEV_SEED_END, "Replay DEV")


def overlaps_eval_range(lo: int, hi: int) -> bool:
    """Does the inclusive seed span [lo, hi] intersect the TEST interval?"""
    return lo <= EVAL_SEED_END and hi >= EVAL_SEED_START


def overlaps_dev_range(lo: int, hi: int) -> bool:
    """Does the inclusive seed span [lo, hi] intersect the DEV interval?"""
    return lo <= DEV_SEED_END and hi >= DEV_SEED_START


def check_training_range(scheme: str, lo: int, hi: int) -> bool:
    """Loud warning when a planned training seed span would touch a reserved
    evaluation interval (TEST or DEV). Returns True when it overlaps. A
    warning, not an error, so existing training workflows keep running — but
    results trained on reserved seeds must not be evaluated on them."""
    hit = []
    if overlaps_eval_range(lo, hi):
        hit.append(f"TEST [{EVAL_SEED_START}, {EVAL_SEED_END}]")
    if overlaps_dev_range(lo, hi):
        hit.append(f"DEV [{DEV_SEED_START}, {DEV_SEED_END}]")
    if hit:
        print(f"WARNING [{scheme}]: planned training seeds {lo}-{hi} overlap "
              f"the reserved evaluation interval(s) {' and '.join(hit)}. A "
              f"model trained on these lobbies must NOT be scored on them — "
              f"choose a different base seed.", file=sys.stderr)
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
