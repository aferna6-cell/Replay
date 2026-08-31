"""Frozen protocol constants and validation for Replay Experiment 3.

This module is deliberately small and side-effect free so both the artifact
generators and tests use one authoritative protocol definition.
"""

from pathlib import Path

from . import seeds

EXPERIMENT_DIR = Path("results/ppo_multiseed_v1")
EXPERIMENT2_DIR = Path("results/ppo_budget_v1")
REPORT_PATH = Path("experiments/ppo_multiseed_replication_v1.md")

TRAINING_SEEDS = (1, 2, 3)
ALL_SEEDS = (0, 1, 2, 3)
ITERATIONS = (0, 40, 80, 160, 320)
EPISODES_PER_ITERATION = 16
SHAPING_HORIZON = 40
GREEDY_GAMES = 1000
MIXED_GAMES = 500
DEV_BASE_SEED = 10_550_000
CORPUS_STATES = 4440
CORPUS_FINGERPRINT = (
    "2ec217b353bd97d7186d5fbe53a7abf019a5036ed53c6a0e888217a77802943e"
)
WARMSTART_PARAMETER_SHA256 = (
    "094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b"
)


def seed_dir(seed: int) -> Path:
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"Experiment 3 trains seeds {TRAINING_SEEDS}, got {seed}")
    return EXPERIMENT_DIR / f"seed_{seed}"


def training_seed_span(seed: int) -> tuple[int, int]:
    """Inclusive BGEnv seed span consumed by one frozen PPO run."""
    return (
        seeds.ppo_episode_seed(seed, 1),
        seeds.ppo_episode_seed(seed, 320 * EPISODES_PER_ITERATION),
    )


def validate_protocol() -> None:
    """Fail closed if a frozen protocol range could touch DEV or TEST."""
    seeds.validate_dev_range(DEV_BASE_SEED, GREEDY_GAMES)
    seeds.validate_dev_range(DEV_BASE_SEED, MIXED_GAMES)
    for seed in TRAINING_SEEDS:
        lo, hi = training_seed_span(seed)
        if seeds.overlaps_dev_range(lo, hi) or seeds.overlaps_eval_range(lo, hi):
            raise ValueError(
                f"training seed {seed} span {lo}-{hi} overlaps DEV or TEST"
            )


def train_command(seed: int) -> list[str]:
    """Exact frozen Experiment 2 recipe for one independent seed."""
    root = seed_dir(seed)
    return [
        "env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "python3", "-m", "ml.train_ppo",
        "--iters", "320",
        "--episodes", str(EPISODES_PER_ITERATION),
        "--seed", str(seed),
        "--shaping", "1.0",
        "--shaping-horizon", str(SHAPING_HORIZON),
        "--from-bc", "ml/policy_bc.pt",
        "--require-from-bc-parameter-sha256", WARMSTART_PARAMETER_SHA256,
        "--out", str(root / "final.pt"),
        "--save-iters", ",".join(str(i) for i in ITERATIONS),
        "--save-dir", str(root / "checkpoints"),
        "--diag-log", str(root / "train_diag.jsonl"),
        "--eval-episodes", "40",
    ]


def eval_command(seed: int, iteration: int, field: str) -> list[str]:
    if seed not in TRAINING_SEEDS or iteration not in ITERATIONS:
        raise ValueError("unknown Experiment 3 seed or iteration")
    games = GREEDY_GAMES if field == "greedy" else MIXED_GAMES
    if field not in ("greedy", "greedy4_random3"):
        raise ValueError(f"unknown Experiment 3 DEV field: {field}")
    root = seed_dir(seed)
    return [
        "python3", "-m", "ml.dev_benchmark",
        "--agent", "policy",
        "--checkpoint", str(root / "checkpoints" / f"iter_{iteration:03d}.pt"),
        "--name", f"ppo-seed{seed}-iter{iteration:03d}",
        "--games", str(games),
        "--seed", str(DEV_BASE_SEED),
        "--field", field,
        "--json-out", str(
            root / "dev" / f"iter{iteration:03d}_vs_{field}.json"
        ),
        "--quiet",
    ]


validate_protocol()
