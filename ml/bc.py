"""Behavior-clone the greedy baseline — the RL warm start (spec §8 Phase 1).

Starting PPO from random weights wastes most of the early samples learning
"play your hand, don't sell your board for nothing." Cloning the scripted
greedy baseline first gives the policy a sane prior in a few CPU-minutes, and
gives the league its first non-scripted member.

  python -m ml.bc --lobbies 150 --epochs 6
"""

import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from hsbg_coach.bg_env import BGEnv, greedy_policy
from hsbg_coach.synergy import load_embeddings
from .env_obs import encode_obs
from .policy_net import PolicyNet, save_policy, as_env_policy
from .rl_common import evaluate_policy, kb_byname
from .tokens import token_dim

_OUT = os.path.join(os.path.dirname(__file__), "policy_bc.pt")


def collect(lobbies: int, emb: Dict, byname: Dict, seed: int = 0) -> List[Tuple]:
    """(arrays, legal, action) demonstrations from the greedy baseline."""
    out = []
    for i in range(lobbies):
        env = BGEnv(seed=seed + i)
        obs = env.reset(seed=seed + i)
        rng = random.Random(seed + i)
        for _ in range(400):
            legal = env.legal_mask(0)
            a = greedy_policy(obs, legal, rng)
            out.append((encode_obs(obs, emb, byname),
                        np.asarray(legal, dtype=np.float32), a))
            obs, _, done, _ = env.step(a)
            if done:
                break
    return out


def collect_dagger(net: PolicyNet, lobbies: int, emb: Dict, byname: Dict,
                   seed: int = 0, mix: float = 0.7) -> List[Tuple]:
    """DAgger round: visit the states the LEARNED policy reaches (that's where
    plain BC compounds its errors) and label every one with the expert's
    action. The expert (greedy_policy) is a pure function of the observation,
    so labels are exact on any state — the ideal DAgger setting."""
    out = []
    for i in range(lobbies):
        env = BGEnv(seed=seed + i)
        obs = env.reset(seed=seed + i)
        rng = random.Random(seed + i)
        for _ in range(400):
            legal = env.legal_mask(0)
            arrays = encode_obs(obs, emb, byname)
            expert = greedy_policy(obs, legal, rng)
            out.append((arrays, np.asarray(legal, dtype=np.float32), expert))
            if rng.random() < mix:
                a, _, _ = net.act(arrays, legal, greedy=True)
            else:
                a = expert
            obs, _, done, _ = env.step(a)
            if done:
                break
    return out


def _action_kind(a: int) -> int:
    from hsbg_coach.bg_env import A_PLAY0, A_SELL0, A_ROLL
    if a < A_PLAY0:
        return 0                                  # buy
    if a < A_SELL0:
        return 1                                  # play
    if a < A_ROLL:
        return 2                                  # sell
    return 3 + (a - A_ROLL)                       # roll / level / freeze / end


def _kind_weights(acts: "torch.Tensor") -> "torch.Tensor":
    """Per-sample weights balancing action KINDS. Rare kinds (level ≈5% of
    decisions) decide games — one mistimed tier-up gets scaling-crushed for
    the rest of the game — but unweighted CE optimizes the common buys/plays
    and shrugs off level mistakes. Inverse-frequency by kind, capped."""
    kinds = torch.tensor([_action_kind(int(a)) for a in acts])
    w = torch.ones(len(acts))
    for k in kinds.unique():
        sel = kinds == k
        w[sel] = float(len(acts)) / (float(sel.sum()) * float(len(kinds.unique())))
    return w.clamp(0.5, 4.0)


def train_bc(demos: List[Tuple], emb: Dict, epochs: int = 6, lr: float = 1e-3,
             batch: int = 256, seed: int = 0, verbose: bool = True) -> PolicyNet:
    torch.manual_seed(seed)
    toks = torch.from_numpy(np.stack([d[0][0] for d in demos]))
    mask = torch.from_numpy(np.stack([d[0][1] for d in demos]))
    zones = torch.from_numpy(np.stack([d[0][2] for d in demos]))
    ctx = torch.from_numpy(np.stack([d[0][3] for d in demos]))
    legal = torch.from_numpy(np.stack([d[1] for d in demos]))
    acts = torch.tensor([d[2] for d in demos], dtype=torch.long)
    weights = _kind_weights(acts)

    net = PolicyNet(token_dim(emb))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = toks.shape[0]
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        correct = 0
        for i in range(0, n, batch):
            ix = perm[i:i + batch]
            opt.zero_grad()
            logits, _ = net(toks[ix], mask[ix], zones[ix], ctx[ix])
            logits = PolicyNet.masked_logits(logits, legal[ix])
            per = F.cross_entropy(logits, acts[ix], reduction="none")
            loss = (per * weights[ix]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            correct += int((logits.argmax(-1) == acts[ix]).sum().item())
        if verbose:
            print(f"epoch {ep}  imitation acc {correct / n:.1%}")
    net.eval()
    return net


def main(argv=None):
    p = argparse.ArgumentParser(description="Behavior-clone the greedy baseline")
    p.add_argument("--lobbies", type=int, default=150)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--dagger-rounds", type=int, default=2,
                   help="DAgger rounds after the initial clone (0 = pure BC)")
    p.add_argument("--dagger-lobbies", type=int, default=80)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=_OUT)
    a = p.parse_args(argv)

    emb = load_embeddings()
    byname = kb_byname()
    print(f"Collecting demonstrations from {a.lobbies} lobbies…")
    demos = collect(a.lobbies, emb, byname, seed=a.seed)
    print(f"  {len(demos)} decisions")
    net = train_bc(demos, emb, epochs=a.epochs, seed=a.seed)

    for rnd in range(a.dagger_rounds):
        print(f"\nDAgger round {rnd + 1}: visiting the learned policy's states…")
        extra = collect_dagger(net, a.dagger_lobbies, emb, byname,
                               seed=a.seed + (rnd + 1) * 10_000)
        demos += extra
        print(f"  +{len(extra)} labeled states (total {len(demos)})")
        net = train_bc(demos, emb, epochs=a.epochs, seed=a.seed)

    save_policy(net, a.out, {"kind": "bc", "demos": len(demos),
                             "dagger_rounds": a.dagger_rounds})
    print(f"Saved -> {a.out}")

    print(f"\nEvaluating vs all-greedy field ({a.eval_episodes} episodes)…")
    avg = evaluate_policy(as_env_policy(net, emb, byname), a.eval_episodes)
    print(f"BC policy avg placement {avg:.2f}  (4.5 = even with the field)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
