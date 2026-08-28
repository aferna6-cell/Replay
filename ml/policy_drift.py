"""Policy-drift diagnostics over a frozen state corpus.

Measures how far PPO checkpoints have moved from their behavior-cloned warm
start, on a fixed corpus of recruit-phase states that is NOT drawn from the
Benchmark v1 TEST interval. The corpus is collected from greedy seat-0
trajectories on a dedicated sub-range of the DEV interval
(``CORPUS_SEED_BASE``), via the same ``ml.bc.collect`` path that produces BC
demonstrations — so every state carries the greedy expert's action label.
The corpus is deterministic given (code, seeds); its SHA-256 fingerprint is
reported so any two runs can prove they scored the same frozen states.

Per checkpoint, against the corpus:
  * expert agreement      — argmax(checkpoint) == greedy expert action
  * warm-start agreement  — argmax(checkpoint) == argmax(reference iter-0)
  * KL from warm start    — mean over states of KL(pi_0 || pi_k) computed on
    legal-action-masked softmax distributions. Direction (documented choice):
    the divergence is weighted by the WARM START's own action distribution,
    i.e. "how much probability mass of the warm start's behavior has the
    checkpoint moved away from".
  * value head            — mean and std of value predictions.

    python -m ml.policy_drift --reference ckpts/iter_000.pt \
        --checkpoints ckpts/iter_*.pt --json-out results/drift.json
"""

import argparse
import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .seeds import DEV_SEED_START

# Diagnostic-corpus sub-range of DEV: disjoint from the dev-eval games at
# DEV_SEED_START+0.. (see ml/dev_benchmark.py), still inside the reserved
# DEV interval, never TEST.
CORPUS_SEED_BASE = DEV_SEED_START + 40_000
CORPUS_LOBBIES = 100


def build_corpus(lobbies: int = CORPUS_LOBBIES,
                 base_seed: int = CORPUS_SEED_BASE,
                 emb: Optional[Dict] = None, byname: Optional[Dict] = None):
    """(states, fingerprint): greedy seat-0 decision states with expert
    labels, plus a SHA-256 over the encoded arrays proving corpus identity."""
    from hsbg_coach.synergy import load_embeddings
    from .bc import collect
    from .rl_common import kb_byname
    emb = emb if emb is not None else load_embeddings()
    byname = byname if byname is not None else kb_byname()
    states = collect(lobbies, emb, byname, seed=base_seed)
    h = hashlib.sha256()
    for arrays, legal, action in states:
        for a in arrays:
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(np.ascontiguousarray(legal).tobytes())
        h.update(int(action).to_bytes(2, "little"))
    return states, h.hexdigest()


def corpus_tensors(states) -> Tuple:
    """Stack the corpus into batched tensors (torch import stays local)."""
    import torch
    toks = torch.from_numpy(np.stack([s[0][0] for s in states]))
    mask = torch.from_numpy(np.stack([s[0][1] for s in states]))
    zones = torch.from_numpy(np.stack([s[0][2] for s in states]))
    ctx = torch.from_numpy(np.stack([s[0][3] for s in states]))
    legal = torch.from_numpy(np.stack([s[1] for s in states]))
    expert = torch.tensor([s[2] for s in states], dtype=torch.long)
    return toks, mask, zones, ctx, legal, expert


def policy_outputs(net, tensors, batch: int = 256):
    """(masked_logits[N,A], values[N]) for a net over the whole corpus."""
    import torch
    from .policy_net import PolicyNet
    toks, mask, zones, ctx, legal, _ = tensors
    net.eval()
    outs, vals = [], []
    with torch.no_grad():
        for i in range(0, toks.shape[0], batch):
            sl = slice(i, i + batch)
            logits, value = net(toks[sl], mask[sl], zones[sl], ctx[sl])
            outs.append(PolicyNet.masked_logits(logits, legal[sl]))
            vals.append(value)
    return torch.cat(outs), torch.cat(vals)


def masked_kl(logits_p, logits_q, legal):
    """Per-state KL(p || q) over legal actions only, from masked logits.

    Both logit tensors must already be -inf on illegal actions (softmax then
    puts exactly zero mass there); the sum is restricted to legal entries so
    the 0 * (-inf - -inf) illegal terms never poison the result.
    """
    import torch
    logp = torch.log_softmax(logits_p, dim=-1)
    logq = torch.log_softmax(logits_q, dim=-1)
    p = torch.exp(logp)
    term = torch.where(legal > 0.5, p * (logp - logq),
                       torch.zeros_like(p))
    return term.sum(dim=-1)


def drift_metrics(logits_k, values_k, logits_ref, tensors) -> Dict:
    """All drift numbers for one checkpoint against the reference (iter 0)."""
    legal, expert = tensors[4], tensors[5]
    acts_k = logits_k.argmax(dim=-1)
    acts_ref = logits_ref.argmax(dim=-1)
    kl = masked_kl(logits_ref, logits_k, legal)
    return {
        "expert_agreement": float((acts_k == expert).float().mean().item()),
        "warmstart_agreement": float((acts_k == acts_ref).float().mean().item()),
        "kl_from_warmstart_mean": float(kl.mean().item()),
        "kl_from_warmstart_p95": float(kl.quantile(0.95).item()),
        "value_mean": float(values_k.mean().item()),
        "value_std": float(values_k.std().item()),
    }


def _sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="ml.policy_drift",
        description="Policy drift of PPO checkpoints from their warm start, "
                    "on a frozen DEV-range state corpus")
    p.add_argument("--reference", required=True,
                   help="the warm-start checkpoint (iteration 0)")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--lobbies", type=int, default=CORPUS_LOBBIES)
    p.add_argument("--corpus-seed", type=int, default=CORPUS_SEED_BASE)
    p.add_argument("--json-out", required=True)
    a = p.parse_args(argv)

    from .policy_net import load_policy
    print(f"Building frozen corpus: {a.lobbies} greedy lobbies from seed "
          f"{a.corpus_seed} (DEV diagnostic range)…")
    states, fingerprint = build_corpus(a.lobbies, a.corpus_seed)
    tensors = corpus_tensors(states)
    print(f"  {len(states)} states, fingerprint {fingerprint[:12]}")

    ref = load_policy(a.reference)
    logits_ref, values_ref = policy_outputs(ref, tensors)
    rows: List[Dict] = []
    for path in a.checkpoints:
        net = load_policy(path)
        logits_k, values_k = policy_outputs(net, tensors)
        row = {"checkpoint": os.path.basename(path),
               "checkpoint_sha256": _sha256(path),
               **drift_metrics(logits_k, values_k, logits_ref, tensors)}
        rows.append(row)
        print(f"  {row['checkpoint']}: expert {row['expert_agreement']:.3f}  "
              f"warmstart {row['warmstart_agreement']:.3f}  "
              f"KL {row['kl_from_warmstart_mean']:.4f}  "
              f"value {row['value_mean']:+.3f}±{row['value_std']:.3f}")

    blob = {
        "corpus": {"lobbies": a.lobbies, "seed_base": a.corpus_seed,
                   "states": len(states), "fingerprint_sha256": fingerprint,
                   "source": "greedy seat-0 trajectories via ml.bc.collect",
                   "split": "dev (diagnostic sub-range; never TEST seeds)"},
        "reference": {"checkpoint": os.path.basename(a.reference),
                      "checkpoint_sha256": _sha256(a.reference)},
        "kl_definition": "mean over states of KL(pi_reference || pi_k) on "
                         "legal-action-masked softmax distributions",
        "checkpoints": rows,
    }
    os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
    with open(a.json_out, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)
    print(f"Saved -> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
