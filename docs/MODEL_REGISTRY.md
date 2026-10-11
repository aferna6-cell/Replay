# NEXT policy model registry

The overlay's NEXT line comes from `ml/policy_net.pt` when `HSBG_NEXT_POLICY` is unset (`0`, `false`, and `off` keep the eval-net advisor). Checkpoints are not in git (`*.pt` is gitignored). This page and [`results/policy_model_registry.json`](../results/policy_model_registry.json) record which file that path should hold.

## Approved: `policy_net_2026-10-07v5.pt` (staged)

Aidan approved this checkpoint on 2026-10-11 at 12:43 PM ET. The laptop was offline at approval time, so the model is staged and the laptop install is pending. On install it becomes `ml/policy_net.pt` and replaces the 2026-09-28c policy model (v4 encoder), which is kept as `ml/policy_net.prev.pt`.

| | |
|---|---|
| File | `policy_net_2026-10-07v5.pt` |
| sha256 | `2fa705f7550956f6fb9154293b6fa89a1d02e704f88c85cc850cd96ea8d50a0c` |
| Encoder | bc-enc-v5 |
| Training | from scratch |
| Code | `pipeline/integration` `6a21999d44248ea346ee9417f33cdc3848b03496` |
| Labels | `v3.14_patch_36_6_3` (BG 36.6.3 only; 1,032 games; 877 train / 155 held-out) |
| Held-out ids sha256 | `25e6214565ff2b307f39f9d8c5f8fd63bc16c2dbb9218153bbaa83e9ffd87db4` |
| Status | approved, staged; laptop install pending |
| Approved | Aidan, 2026-10-11, 12:43 PM ET |
| Replaces | 2026-09-28c policy model (v4 encoder), kept as `ml/policy_net.prev.pt` on install |

### Gate: PASS

- model_coarse top-1 36.3 vs advisor_hand_lines 17.7 (eval_net 9.0), n=55,156
- hero_power 65.2 vs 61.7 (n=958; 0 out-of-vocab)
- Sylvanas HP 87.4 vs 82.0 (n=56)
- freeze 14.6 vs 39.0 (exempt)

### Known gap

E.T.C. `BG26_HERO_100` is not in `HERO_VOCAB`. That is fixed separately for the next encoder version.

## Previous: 2026-09-28c

Encoder bc-enc-v4. This is the checkpoint on the laptop at `ml/policy_net.pt` until the staged v5 file is installed. On that install it is kept as `ml/policy_net.prev.pt`. `scripts/policy_rollback.ps1` copies that previous file back over `ml/policy_net.pt`.
