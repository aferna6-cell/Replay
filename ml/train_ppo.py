"""PPO with a self-play league — Phase 1/2 of the RL spec.

  python -m ml.train_ppo --iters 40 --episodes 16

The backbone the spec recommends (§6): PPO on the Phase 0 env, warm-started
from the behavior-cloned policy when `ml/policy_bc.pt` exists, playing against
a *league* — scripted greedy/random plus snapshots of past selves — rather
than only the latest self (the AlphaStar lesson: leagues prevent strategy
collapse). Reward is zero-mean placement; a small on-pace shaping term anneals
to zero over training (§5).

Gate to report: average placement vs the all-greedy field. Below 4.5 beats the
field; the scripted-greedy baseline itself sits at ~4.5 by construction.
"""

import argparse
import copy
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from hsbg_coach.synergy import load_embeddings
from . import seeds
from .policy_net import PolicyNet, save_policy, load_policy, as_env_policy
from .rl_common import rollout, mixed_field, evaluate_policy, kb_byname
from .tokens import token_dim

_OUT = os.path.join(os.path.dirname(__file__), "policy_ppo.pt")
_BC = os.path.join(os.path.dirname(__file__), "policy_bc.pt")

GAMMA = 0.999
LAM = 0.95
CLIP = 0.2
ENTROPY = 0.01
VALUE_COEF = 0.5
PPO_EPOCHS = 4
LEAGUE_EVERY = 8
LEAGUE_MAX = 5


def diag_stats(old_logp, logp, clip: float = CLIP):
    """(approx_kl, clip_fraction) for one minibatch — pure observation.

    approx_kl is Schulman's cheap estimator of KL(old || new),
    ``mean(old_logp - logp)``; clip_fraction is the share of samples whose
    probability ratio left the PPO trust region ``|r - 1| > clip``. Neither
    feeds back into the loss, the gradients, or the RNG.
    """
    ratio = torch.exp(logp - old_logp)
    return (float((old_logp - logp).mean().item()),
            float(((ratio - 1.0).abs() > clip).float().mean().item()))


def rl_signal(adv_raw, ret, values, placements, shaping_r, terminal_r) -> dict:
    """Does the policy receive variation that can tell good decisions from
    bad ones? Measured on RAW advantages, before PPO's per-batch
    normalization (which rescales any signal, however weak, to unit std).

    * ``adv_*`` — GAE advantages exactly as ``_gae`` produced them.
    * ``adv_frac_positive/zero/negative`` — sign split; a healthy batch
      separates actions in both directions. "zero" means |a| < 1e-8.
    * ``value_explained_variance`` — 1 - Var(returns - values) / Var(returns).
      1.0 = the value head explains returns perfectly, 0.0 = no better than
      predicting the mean, negative = worse than the mean.
    * ``shaping_reward_sum`` / ``terminal_reward_sum`` — the two reward
      sources separated, so their relative contribution is visible as
      shaping anneals away.
    """
    adv = np.asarray(adv_raw, dtype=np.float64)
    ret = np.asarray(ret, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    resid = ret - val
    ret_var = float(ret.var())
    return {
        "adv_mean": float(adv.mean()),
        "adv_std": float(adv.std()),
        "adv_mean_abs": float(np.abs(adv).mean()),
        "adv_frac_positive": float((adv > 1e-8).mean()),
        "adv_frac_zero": float((np.abs(adv) <= 1e-8).mean()),
        "adv_frac_negative": float((adv < -1e-8).mean()),
        "return_mean": float(ret.mean()),
        "return_std": float(ret.std()),
        "value_pred_mean": float(val.mean()),
        "value_pred_std": float(val.std()),
        "value_explained_variance": (float(1.0 - resid.var() / ret_var)
                                     if ret_var > 1e-12 else None),
        "placement_mean": float(np.mean(placements)),
        "placement_std": float(np.std(placements)),
        "placement_distinct": len(set(placements)),
        "shaping_reward_sum": float(np.sum(shaping_r)),
        "terminal_reward_sum": float(np.sum(terminal_r)),
    }


def _gae(rewards: List[float], values: List[float]) -> np.ndarray:
    adv = np.zeros(len(rewards), dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nv = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + GAMMA * nv - values[t]
        last = delta + GAMMA * LAM * last
        adv[t] = last
    return adv


def _update(net, opt, batch) -> dict:
    """One PPO update pass. The extra stats (approx KL, clip fraction, grad
    norm) are pure observations on tensors the optimization already computes
    — no additional RNG draws, no change to gradients or step order."""
    toks, mask, zones, ctx, legal, acts, old_logp, adv, ret = batch
    stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "approx_kl": 0.0,
             "clip_frac": 0.0, "grad_norm": 0.0, "batches": 0}
    n = toks.shape[0]
    for _ in range(PPO_EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            ix = perm[i:i + 256]
            logits, value = net(toks[ix], mask[ix], zones[ix], ctx[ix])
            logits = PolicyNet.masked_logits(logits, legal[ix])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(acts[ix])
            ratio = torch.exp(logp - old_logp[ix])
            a = adv[ix]
            pi_loss = -torch.min(ratio * a,
                                 torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a).mean()
            v_loss = F.mse_loss(value, ret[ix])
            ent = dist.entropy().mean()
            loss = pi_loss + VALUE_COEF * v_loss - ENTROPY * ent
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats["pi"] += float(pi_loss.item())
            stats["v"] += float(v_loss.item())
            stats["ent"] += float(ent.item())
            with torch.no_grad():
                kl, cf = diag_stats(old_logp[ix], logp)
                stats["approx_kl"] += kl
                stats["clip_frac"] += cf
            stats["grad_norm"] += float(grad_norm.item())
            stats["batches"] += 1
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description="PPO + league self-play")
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--episodes", type=int, default=16, help="episodes per iter")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shaping", type=float, default=1.0,
                   help="initial on-pace shaping weight (anneals to 0)")
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--out", default=_OUT)
    p.add_argument("--from-bc", default=_BC)
    p.add_argument("--require-from-bc-parameter-sha256",
                   help="fail before training unless --from-bc exists and has "
                        "this filename-independent parameter fingerprint; "
                        "a reproducibility guard that does not alter training")
    p.add_argument("--save-iters", default="",
                   help="comma-separated iteration numbers to snapshot for "
                        "diagnostics (0 = the exact warm-start weights before "
                        "any PPO update); requires --save-dir. Snapshotting "
                        "never perturbs training RNG or behavior.")
    p.add_argument("--save-dir", help="directory for --save-iters snapshots")
    p.add_argument("--diag-log",
                   help="append per-iteration optimization diagnostics "
                        "(losses, entropy, approx KL, clip fraction, grad "
                        "norm, rollout placement, shaping, league size, "
                        "steps, and raw advantage/return/value signal) as "
                        "JSON lines to this file")
    p.add_argument("--shaping-horizon", type=int, default=None,
                   help="anneal the shaping weight against this iteration "
                        "count instead of --iters. Defaults to --iters (the "
                        "shipped behavior). Set it to pin the ORIGINAL "
                        "schedule when extending a run: --iters 320 "
                        "--shaping-horizon 40 reproduces the 40-iteration "
                        "schedule for iterations 1-40 and holds shaping at 0 "
                        "afterwards, so training budget is the only variable.")
    a = p.parse_args(argv)
    horizon = a.shaping_horizon if a.shaping_horizon else a.iters

    save_iters = {int(x) for x in a.save_iters.split(",") if x.strip()}
    if save_iters and not a.save_dir:
        p.error("--save-iters requires --save-dir")

    def _snapshot(iteration: int) -> None:
        if iteration in save_iters:
            os.makedirs(a.save_dir, exist_ok=True)
            save_policy(net, os.path.join(a.save_dir,
                                          f"iter_{iteration:03d}.pt"),
                        {"kind": "ppo-diagnostic", "iter": iteration,
                         "seed": a.seed})

    def _diag(record: dict) -> None:
        if a.diag_log:
            import json
            with open(a.diag_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    emb = load_embeddings()
    byname = kb_byname()

    if a.require_from_bc_parameter_sha256:
        if not os.path.isfile(a.from_bc):
            p.error(f"required warm-start checkpoint does not exist: {a.from_bc}")
        from .model_fingerprint import checkpoint_parameter_sha256
        actual = checkpoint_parameter_sha256(a.from_bc)
        if actual != a.require_from_bc_parameter_sha256:
            p.error("warm-start parameter SHA256 mismatch: "
                    f"expected {a.require_from_bc_parameter_sha256}, got {actual}")

    if os.path.isfile(a.from_bc):
        net = load_policy(a.from_bc)
        print(f"Warm-started from {a.from_bc}")
    else:
        net = PolicyNet(token_dim(emb))
        print("No BC checkpoint — starting from scratch "
              "(run `python -m ml.bc` first for a better start)")
    net.train()
    _snapshot(0)             # exact warm-start weights, before any PPO update
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    league: List = []

    def policy_step(arrays, legal):
        return net.act(arrays, legal, greedy=False)

    seeds.check_training_range(
        "ml.train_ppo", seeds.ppo_episode_seed(a.seed, 1),
        seeds.ppo_episode_seed(a.seed, a.iters * a.episodes))
    ep_index = 0
    for it in range(a.iters):
        anneal = max(0.0, 1.0 - it / max(1, horizon * 0.7))
        shaping = a.shaping * anneal
        trajs = []
        for e in range(a.episodes):
            ep_index += 1
            trajs.append(rollout(policy_step,
                                 seeds.ppo_episode_seed(a.seed, ep_index),
                                 opponents=mixed_field(rng, league),
                                 emb=emb, byname=byname, shaping=shaping))
        placements = [t["placement"] for t in trajs]

        toks = torch.from_numpy(np.concatenate([np.stack(t["tokens"]) for t in trajs]))
        mask = torch.from_numpy(np.concatenate([np.stack(t["mask"]) for t in trajs]))
        zones = torch.from_numpy(np.concatenate([np.stack(t["zones"]) for t in trajs]))
        ctx = torch.from_numpy(np.concatenate([np.stack(t["ctx"]) for t in trajs]))
        legal = torch.from_numpy(np.concatenate([np.stack(t["legal"]) for t in trajs]))
        acts = torch.tensor([x for t in trajs for x in t["action"]], dtype=torch.long)
        old_logp = torch.tensor([x for t in trajs for x in t["logp"]])
        advs, rets = [], []
        for t in trajs:
            adv = _gae(t["reward"], t["value"])
            advs.append(adv)
            rets.append(adv + np.asarray(t["value"], dtype=np.float32))
        adv = torch.from_numpy(np.concatenate(advs))
        ret = torch.from_numpy(np.concatenate(rets))
        # RL-signal diagnostics on the RAW advantages, before normalization.
        signal = rl_signal(
            np.concatenate(advs), np.concatenate(rets),
            [v for t in trajs for v in t["value"]], placements,
            [r for t in trajs for r in t["shaping_reward"]],
            [r for t in trajs for r in t["terminal_reward"]])
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        stats = _update(net, opt,
                        (toks, mask, zones, ctx, legal, acts, old_logp, adv, ret))
        print(f"iter {it:3d}  avg placement {np.mean(placements):.2f}  "
              f"steps {toks.shape[0]:4d}  shaping {shaping:.2f}  "
              f"league {len(league)}")
        nb = max(1, stats["batches"])
        _diag({"iter": it + 1,
               "rollout_avg_placement": float(np.mean(placements)),
               "steps": int(toks.shape[0]), "shaping": float(shaping),
               "league_size": len(league),
               "pi_loss": stats["pi"] / nb, "v_loss": stats["v"] / nb,
               "entropy": stats["ent"] / nb,
               "approx_kl": stats["approx_kl"] / nb,
               "clip_frac": stats["clip_frac"] / nb,
               "grad_norm": stats["grad_norm"] / nb,
               "minibatches": stats["batches"], **signal})
        _snapshot(it + 1)

        if (it + 1) % LEAGUE_EVERY == 0:
            frozen = copy.deepcopy(net)
            frozen.eval()
            league.append(as_env_policy(frozen, emb, byname))
            if len(league) > LEAGUE_MAX:
                league.pop(0)
            save_policy(net, a.out, {"kind": "ppo", "iter": it + 1})

    net.eval()
    save_policy(net, a.out, {"kind": "ppo", "iters": a.iters})
    print(f"\nSaved -> {a.out}")
    print(f"Evaluating vs all-greedy field ({a.eval_episodes} episodes)…")
    avg = evaluate_policy(as_env_policy(net, emb, byname), a.eval_episodes)
    print(f"PPO policy avg placement vs greedy field: {avg:.2f}  "
          f"(<4.5 beats the field)")
    from hsbg_coach.bg_env import random_policy
    avg_r = evaluate_policy(as_env_policy(net, emb, byname), a.eval_episodes,
                            field=[random_policy] * 7)
    print(f"PPO policy avg placement vs random field: {avg_r:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
