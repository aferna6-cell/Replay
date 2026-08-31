"""Diagnose a checkpoint that FAILS the frozen DEV evaluation protocol.

    python scripts/ppo_multiseed_protocol_failure.py --seed 1 --iteration 320

The Benchmark v1 / DEV machinery refuses to score an episode that does not
terminate within MAX_DECISIONS (a silent 8th place would corrupt the
numbers), so a checkpoint whose deterministic argmax play loops forever in
any lobby of the DEV block has NO defined score under the frozen protocol.
This tool documents that failure instead of hiding it: it replays every game
of the protocol individually, records which game seeds are non-terminating
(and on which action the policy loops), and stores the placements of the
games that did finish as a clearly-labeled DIAGNOSTIC — explicitly NOT a
benchmark or DEV result, never comparable to scored checkpoints.

Output: ``<seed_dir>/dev/iter{IT:03d}_vs_{field}.protocol_failure.json``
for each requested field. The frozen protocol itself is unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.benchmark import (BenchmarkIntegrityError, make_agent,  # noqa: E402
                          run_game)
from ml.dev_benchmark import dev_field_seats, field_composition  # noqa: E402
from ml.model_fingerprint import checkpoint_fingerprint  # noqa: E402
from ml.multiseed_analysis import (DEV_EVAL_BASE, DEV_EVAL_GAMES,  # noqa: E402
                                   MIXED_FIELD, MIXED_GAMES, seed_dir,
                                   write_json)
from ml.rl_common import MAX_DECISIONS  # noqa: E402
from ml.seeds import validate_dev_range  # noqa: E402


def _loop_action(checkpoint: str, seats, game_seed: int):
    """Which action the argmax policy repeats when the game never ends."""
    import random
    from hsbg_coach.bg_env import BGEnv
    agent = make_agent("policy", checkpoint, "loop-probe")
    env = BGEnv(seed=game_seed, opponent_policies=list(seats))
    obs = env.reset(seed=game_seed)
    rng = random.Random(game_seed)
    acts = []
    for _ in range(MAX_DECISIONS):
        mask = env.legal_mask(0)
        a = agent.policy(obs, mask, rng)
        acts.append(int(a))
        obs, _, done, _ = env.step(a)
        if done:
            return None
    tail = Counter(acts[-50:])
    action, count = tail.most_common(1)[0]
    return {"dominant_tail_action": action, "tail_share": count / 50,
            "final_turn": int(obs["turn"])}


def diagnose(seed: int, iteration: int, directory: str | None = None) -> dict:
    directory = directory or seed_dir(seed)
    checkpoint = os.path.join(directory, "checkpoints",
                              f"iter_{iteration:03d}.pt")
    fp = checkpoint_fingerprint(checkpoint)
    out = {}
    for field, games in (("greedy", DEV_EVAL_GAMES),
                         (MIXED_FIELD, MIXED_GAMES)):
        validate_dev_range(DEV_EVAL_BASE, games)
        seats = dev_field_seats(field)
        agent = make_agent("policy", checkpoint, f"iter{iteration:03d}")
        completed, non_terminating = [], []
        for i in range(games):
            game_seed = DEV_EVAL_BASE + i
            try:
                res = run_game(agent, seats, game_seed)
                completed.append({"seed": game_seed,
                                  "placement": res["placement"]})
            except BenchmarkIntegrityError:
                non_terminating.append(game_seed)
        loops = [{"seed": s, **(_loop_action(checkpoint, seats, s) or {})}
                 for s in non_terminating[:5]]        # probe a few
        blob = {
            "kind": "PROTOCOL FAILURE DIAGNOSTIC — NOT a benchmark or DEV "
                    "result",
            "why": "the frozen DEV protocol refuses to score episodes that "
                   "do not terminate within MAX_DECISIONS, so this "
                   "checkpoint has NO defined DEV score; the numbers below "
                   "only characterize the failure and must never be "
                   "compared against scored checkpoints",
            "training_seed": seed,
            "ppo_iteration": iteration,
            "cumulative_training_episodes": iteration * 16,
            "checkpoint": os.path.basename(checkpoint),
            **fp,
            "field": field,
            "field_composition": field_composition(field),
            "games_attempted": games,
            "seed_range": [DEV_EVAL_BASE, DEV_EVAL_BASE + games - 1],
            "max_decisions": MAX_DECISIONS,
            "n_completed": len(completed),
            "n_non_terminating": len(non_terminating),
            "non_terminating_game_seeds": non_terminating,
            "loop_probes": loops,
            "completed_games_diagnostic": {
                "note": "placements of the games that DID finish — "
                        "diagnostic only, biased by the missing games",
                "avg_placement": (sum(c["placement"] for c in completed)
                                  / len(completed)) if completed else None,
                "placements": [c["placement"] for c in completed],
            },
        }
        path = os.path.join(directory, "dev",
                            f"iter{iteration:03d}_vs_{field}"
                            f".protocol_failure.json")
        write_json(path, blob)
        print(f"{field}: {len(non_terminating)}/{games} games "
              f"non-terminating -> {path}")
        out[field] = blob
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="document a DEV protocol failure")
    p.add_argument("--seed", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--iteration", type=int, default=320)
    p.add_argument("--dir")
    a = p.parse_args(argv)
    diagnose(a.seed, a.iteration, a.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
