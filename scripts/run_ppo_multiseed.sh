#!/usr/bin/env bash
# Sequential Experiment 3 training + DEV eval for seeds 1, 2, 3.
# Frozen Experiment 2 recipe. Do not tune. Do not touch TEST.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export PATH="${HOME}/.local/bin:${PATH}"

python3 - <<'PY'
from ml.model_fingerprint import checkpoint_parameter_sha256
from ml.multiseed_analysis import (WARM_START_PARAMETER_SHA256,
                                   assert_training_seeds_isolated,
                                   assert_warmstart_hash)
assert_warmstart_hash(checkpoint_parameter_sha256("ml/policy_bc.pt"))
assert_training_seeds_isolated([1, 2, 3])
print("warm-start ok", WARM_START_PARAMETER_SHA256)
print("training seeds 1-3 isolated from DEV/TEST")
PY

for s in 1 2 3; do
  d="results/ppo_multiseed_v1/seed_${s}"
  mkdir -p "${d}/checkpoints" "${d}/dev"
  echo "======== TRAIN SEED ${s} $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"
  python3 scripts/ppo_multiseed_train.py --seed "${s}" 2>&1 | tee "${d}/train.log"
  echo "======== EVAL SEED ${s} $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"
  python3 scripts/ppo_multiseed_eval.py --seed "${s}" --quiet 2>&1 | tee "${d}/eval.log"
  python3 scripts/ppo_multiseed_seed_report.py --seed "${s}" 2>&1 | tee "${d}/report.log"
  echo "======== SEED ${s} COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"
done

echo "======== AGGREGATE $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"
python3 scripts/ppo_multiseed_aggregate.py
python3 scripts/ppo_multiseed_manifest.py
echo "======== ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ========"
