"""Self-play policy: learn to *act* in the economy game, not just evaluate.

A policy network maps an economy state -> P(level vs tempo this turn), trained by
REINFORCE: all 8 players sample from the current policy, the lobby is played out,
and each decision is reinforced in proportion to the finish it led to (reward =
how much better than 4.5 you placed), with a mean-reward baseline for variance.

This is the genuine self-play RL loop (scoped to the economy layer): the policy
improves itself by playing against copies of itself. We then check it (a) beats
the heuristic field and (b) rediscovers sensible strategy (level when behind).
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .econ_env import simulate_lobby, features
from hsbg_coach.pace import load_pace

ACTIONS = ["level", "tempo"]


class PolicyNet(nn.Module):
    def __init__(self, n_in: int = 6, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, len(ACTIONS)),
        )

    def forward(self, x):
        return self.net(x)


def make_decider(net: PolicyNet, greedy: bool = False):
    """A deciders[i] callable: features -> (intent, action_idx)."""
    def decide(feat):
        with torch.no_grad():
            logits = net(torch.tensor([feat], dtype=torch.float32))
            probs = F.softmax(logits, dim=-1)[0]
            a = int(torch.argmax(probs)) if greedy else int(torch.multinomial(probs, 1))
        return ACTIONS[a], a
    return decide


def reward(placement: int) -> float:
    return (4.5 - placement) / 3.5            # 1st -> +1, 8th -> -1


def evaluate(net: PolicyNet, pace, n: int = 200, seed: int = 50000) -> float:
    """Avg placement of one greedy-policy player vs 7 heuristic players (<4.5 = wins)."""
    deciders = [make_decider(net, greedy=True)] + [None] * 7
    places = []
    for i in range(n):
        players = simulate_lobby(pace, seed=seed + i, deciders=deciders)
        places.append(players[0].placement)
    return float(np.mean(places))


def train(iters: int = 60, lobbies: int = 64, lr: float = 0.01, hidden: int = 32,
          seed: int = 0, verbose: bool = True) -> Tuple[PolicyNet, dict]:
    torch.manual_seed(seed)
    net = PolicyNet(6, hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    pace = load_pace()
    history = []
    for it in range(iters):
        feats, acts, rets = [], [], []
        decider = make_decider(net)               # sampling policy, shared by all 8
        for g in range(lobbies):
            players = simulate_lobby(pace, seed=seed + it * lobbies + g,
                                     deciders=[decider] * 8)
            for p in players:
                R = reward(p.placement)
                for feat, aidx in p.actions:
                    feats.append(feat)
                    acts.append(aidx)
                    rets.append(R)
        Ft = torch.tensor(feats, dtype=torch.float32)
        At = torch.tensor(acts, dtype=torch.long)
        Rt = torch.tensor(rets, dtype=torch.float32)
        adv = Rt - Rt.mean()                      # baseline-subtracted advantage
        logp = F.log_softmax(net(Ft), dim=-1).gather(1, At.unsqueeze(1)).squeeze(1)
        loss = -(logp * adv).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (it % 10 == 0 or it == iters - 1):
            ev = evaluate(net, pace, n=120)
            history.append((it, ev))
            print(f"iter {it:3d}  vs-field avg place {ev:.2f}")
    return net, {"history": history, "final": evaluate(net, pace, n=300)}


def save(net: PolicyNet, path: str):
    torch.save({"state": net.state_dict(), "n_in": net.net[0].in_features}, path)


def load(path: str) -> PolicyNet:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyNet(blob["n_in"])
    net.load_state_dict(blob["state"])
    return net


def recommend_intent(turn, tier, strength, ratio, hp, players_left, net) -> Tuple[str, float]:
    """The policy's recommended this-turn intent + its probability."""
    with torch.no_grad():
        probs = F.softmax(net(torch.tensor(
            [features(turn, tier, strength, ratio, hp, players_left)],
            dtype=torch.float32)), dim=-1)[0]
    a = int(torch.argmax(probs))
    return ACTIONS[a], float(probs[a])
