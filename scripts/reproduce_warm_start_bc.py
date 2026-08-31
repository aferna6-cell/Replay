"""Reproduce the Experiment 2 BC + DAgger warm-start checkpoint.

The historical ``ml/policy_bc.pt`` is gitignored. On CPU torch, matching its
``parameter_sha256`` requires ``torch.use_deterministic_algorithms(True)``
*without* forcing single-threaded BLAS (single-thread yields a different,
still-deterministic trajectory). Verifies against the Exp2 manifest hash
before writing.

    PYTHONHASHSEED=0 python scripts/reproduce_warm_start_bc.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import random
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsbg_coach.synergy import load_embeddings
from ml import seeds
from ml.bc import collect, collect_dagger, train_bc
from ml.model_fingerprint import checkpoint_fingerprint, parameter_sha256
from ml.policy_net import save_policy
from ml.rl_common import kb_byname

EXPECTED_PARAMETER_SHA256 = (
    "094417bdcaa7af6298c5239bbefdc8340b1579601ebb36c6380aea11246d473b"
)
OUT = "ml/policy_bc.pt"


def main() -> int:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    emb = load_embeddings()
    byname = kb_byname()
    print("Collecting demonstrations from 150 lobbies…")
    demos = collect(150, emb, byname, seed=0)
    print(f"  {len(demos)} decisions")
    if len(demos) != 7319:
        print(f"WARNING: expected 7319 demos, got {len(demos)}", file=sys.stderr)

    net = train_bc(demos, emb, epochs=6, seed=0)
    for rnd in range(2):
        print(f"\nDAgger round {rnd + 1}: visiting the learned policy's states…")
        extra = collect_dagger(net, 80, emb, byname,
                               seed=seeds.dagger_round_base(0, rnd + 1))
        demos = demos + extra
        print(f"  +{len(extra)} labeled states (total {len(demos)})")
        net = train_bc(demos, emb, epochs=6, seed=0)

    h = parameter_sha256(net.state_dict())
    print(f"parameter_sha256 = {h}")
    if h != EXPECTED_PARAMETER_SHA256:
        print("REFUSING to write: hash does not match Experiment 2 warm start",
              file=sys.stderr)
        print(f"  expected {EXPECTED_PARAMETER_SHA256}", file=sys.stderr)
        return 1

    save_policy(net, OUT, {
        "kind": "bc+dagger",
        "demos": len(demos),
        "dagger_rounds": 2,
        "deterministic_torch": True,
        "experiment2_warm_start_parameter_sha256": EXPECTED_PARAMETER_SHA256,
    })
    fp = checkpoint_fingerprint(OUT)
    print(f"Saved -> {OUT}")
    print(f"fingerprint {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
