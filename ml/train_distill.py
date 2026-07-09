"""Search-guided distillation — the model trains by simulating games.

The AlphaZero-shaped improvement loop, adapted to the Phase 0 env:

  1. The student policy plays simulated lobbies (its own states, where its
     mistakes actually live — the DAgger insight).
  2. The TEACHER — beam search over whole turns, scored by the set-net board
     brain (`ml/search_expert.py`) — labels every visited state with the move
     it would make.
  3. The student is retrained to match; repeat. Each round the student gets
     closer to "search-quality play at network speed."

  python -m ml.train_distill --rounds 4 --lobbies 40

Every round reports the numbers that matter: average placement vs the
all-greedy field, top-4 rate, final tier, and HOW BIG ITS BOARDS GET (final
total stats, absolute and relative to the best opponent board). The teacher
itself is evaluated first — it is the student's ceiling, and if the teacher
can't beat greedy, the env (not the student) is the bottleneck.

The distilled checkpoint drops into PPO as the warm start:
  python -m ml.train_ppo --from-bc ml/policy_distill.pt
"""

import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np

from hsbg_coach.bg_env import BGEnv, greedy_policy
from hsbg_coach.synergy import load_embeddings
from .bc import train_bc
from .env_obs import encode_obs
from .policy_net import PolicyNet, save_policy, load_policy, as_env_policy
from .rl_common import evaluate_detailed, kb_byname
from .search_expert import SearchExpert
from .tokens import token_dim

_OUT = os.path.join(os.path.dirname(__file__), "policy_distill.pt")
_BC = os.path.join(os.path.dirname(__file__), "policy_bc.pt")


def _fmt(name: str, m: Dict) -> str:
    return (f"{name:22s} placement {m['placement']:.2f}  top4 {m['top4']:.0%}  "
            f"final board {m['final_board_stats']:.0f} stats "
            f"({m['vs_best_opp_board']:.2f}x best opp)  tier {m['final_tier']:.1f}")


def collect_round(student: PolicyNet, expert: SearchExpert, lobbies: int,
                  emb: Dict, byname: Dict, seed: int, mix: float) -> List[Tuple]:
    """Roll lobbies with a student/expert action mixture; label EVERY visited
    state with the expert's action. mix = P(student drives the next step)."""
    out = []
    for i in range(lobbies):
        env = BGEnv(seed=seed + i)
        obs = env.reset(seed=seed + i)
        rng = random.Random(seed + i)
        for _ in range(400):
            legal = env.legal_mask(0)
            arrays = encode_obs(obs, emb, byname)
            label = expert(obs, legal, rng)
            out.append((arrays, np.asarray(legal, dtype=np.float32), label))
            if rng.random() < mix:
                a, _, _ = student.act(arrays, legal, greedy=True)
            else:
                a = label
            obs, _, done, _ = env.step(a)
            if done:
                break
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Distill the beam-search teacher")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--lobbies", type=int, default=40, help="lobbies per round")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--teacher", choices=["greedy", "search"], default="greedy",
                   help="greedy: the env's near-optimal curve-following "
                        "stat-max baseline (the env's growth model is tuned "
                        "around it — measured stronger than the search here). "
                        "search: the beam-search+set-net advisor (real-game "
                        "knowledge; the right teacher once the env models "
                        "real card effects).")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--teacher-eval-episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=_OUT)
    a = p.parse_args(argv)

    emb = load_embeddings()
    byname = kb_byname()
    if a.teacher == "search":
        # env_mode: inside the Phase 0 env the teacher must optimize the
        # env's actual dynamics (compounding board scaling), not the real
        # meta — the knowledge-laden search churns comps and loses here.
        expert = SearchExpert(beam=a.beam, depth=a.depth, env_mode=True)
    else:
        expert = greedy_policy

    print(f"Evaluating the TEACHER ({a.teacher}) vs all-greedy…")
    tm = evaluate_detailed(expert, a.teacher_eval_episodes, seed=8800)
    print(_fmt("teacher", tm))
    gm = evaluate_detailed(greedy_policy, a.eval_episodes, seed=8800)
    print(_fmt("greedy baseline", gm))

    if os.path.isfile(_BC):
        student = load_policy(_BC)
        print(f"\nStudent warm-started from {_BC}")
    else:
        student = PolicyNet(token_dim(emb))
        print("\nStudent starts untrained (no BC checkpoint found)")

    dataset: List[Tuple] = []
    for rnd in range(a.rounds):
        # Early rounds lean on the expert driving (good states to imitate);
        # later rounds lean on the student driving (its own mistakes to fix).
        mix = min(0.8, 0.3 + 0.2 * rnd)
        print(f"\n=== Round {rnd + 1}/{a.rounds} — {a.lobbies} lobbies, "
              f"student drives {mix:.0%} of steps ===")
        fresh = collect_round(student, expert, a.lobbies, emb, byname,
                              seed=a.seed + rnd * 50_000, mix=mix)
        dataset += fresh
        print(f"  +{len(fresh)} labeled decisions (total {len(dataset)})")
        student = train_bc(dataset, emb, epochs=a.epochs, seed=a.seed,
                           verbose=False)
        sm = evaluate_detailed(as_env_policy(student, emb, byname),
                               a.eval_episodes, seed=8800)
        print(_fmt(f"student r{rnd + 1}", sm))
        save_policy(student, a.out, {"kind": "distill", "round": rnd + 1,
                                     "decisions": len(dataset)})

    print(f"\nSaved -> {a.out}")
    print("Next: python -m ml.train_ppo --from-bc", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
