# Local HSReplay-only eval-net train (36.6.1)

Skips Firestone live fetch and VOD trajectories. Uses the wired train corpus:

- `data/hsreplay/36.6.1` — **11 kept intact comps + 5 synth v2** (dropped/naga zero-weighted)
- heroes / trinkets / spells as **SUPPORT** examples (see `ml/hsreplay_snapshot_dataset.py`)
- `data/train_perfect_thin150_clean` — scrubbed perfect boards (72 in-pool only)
- T7 minions: useful adds, **NOT required**
- Naga out. No Firestone. No VODs.

Audit: `data/TRAIN_META_AUDIT.md`

```powershell
cd C:\Users\aidan\Replay
.\.venv\Scripts\python.exe -m ml.train_eval_net `
  --hsreplay-snapshot data/hsreplay/36.6.1 `
  --trajectories data/train_perfect_thin150_clean `
  --epochs 40 `
  --out results/eval_net_hsreplay_36_6_1.pt
Copy-Item results\eval_net_hsreplay_36_6_1.pt ml\eval_net.pt -Force
.\.venv\Scripts\python.exe -m hsbg_coach watch --overlay
```

One-liner:

```powershell
cd C:\Users\aidan\Replay; .\.venv\Scripts\python.exe -m ml.train_eval_net --hsreplay-snapshot data/hsreplay/36.6.1 --trajectories data/train_perfect_thin150_clean --epochs 40 --out results/eval_net_hsreplay_36_6_1.pt; Copy-Item results\eval_net_hsreplay_36_6_1.pt ml\eval_net.pt -Force
```

## Latest run (2026-09-23 ET)

| | |
|--|--|
| Population boards | 200 (64 comps + 36 heroes + 28 trinkets + 72 spells support) |
| Thin150 clean | 72 |
| Train / val | 230 / 42 |
| Features / heroes | 70 / 37 |
| **val MAE** | **0.617** |
| **val Pearson r** | **0.805** |
| Checkpoint | 
esults/eval_net_hsreplay_36_6_1.pt (copied to ml/eval_net.pt) |
| Log | 
esults/train_eval_net_36_6_1.log |

## Weight note (2026-09-23)
Comps 4x (highest); heroes/trinkets/spells support **2x**. Re-run train after this bump.
