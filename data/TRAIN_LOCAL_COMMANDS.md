# TRAIN_LOCAL_COMMANDS — 36.6.1 wired corpus

Run from repo root. Python: `.\.venv\Scripts\python.exe`

## Train (canonical)

```powershell
cd C:\Users\aidan\Replay
New-Item -ItemType Directory -Force -Path results | Out-Null
.\.venv\Scripts\python.exe -m ml.train_eval_net --hsreplay-snapshot data/hsreplay/36.6.1 --trajectories data/train_perfect_thin150_clean --epochs 40 --out results/eval_net_hsreplay_36_6_1.pt
Copy-Item results\eval_net_hsreplay_36_6_1.pt ml\eval_net.pt -Force
```

## Background train (if interactive session is short)

```powershell
cd C:\Users\aidan\Replay
New-Item -ItemType Directory -Force -Path results | Out-Null
$p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m","ml.train_eval_net","--hsreplay-snapshot","data/hsreplay/36.6.1","--trajectories","data/train_perfect_thin150_clean","--epochs","40","--out","results/eval_net_hsreplay_36_6_1.pt" -RedirectStandardOutput "results\train_eval_net_36_6_1.log" -RedirectStandardError "results\train_eval_net_36_6_1.err" -PassThru -NoNewWindow
"PID=$($p.Id)" | Tee-Object -FilePath results\train_eval_net_36_6_1.pid
Get-Content results\train_eval_net_36_6_1.log -Wait
```

After finish:

```powershell
Copy-Item results\eval_net_hsreplay_36_6_1.pt ml\eval_net.pt -Force
```

## Smoke: dataset build only

```powershell
.\.venv\Scripts\python.exe -c "from ml.hsreplay_snapshot_dataset import build_hsreplay_snapshot_examples; ex=build_hsreplay_snapshot_examples('data/hsreplay/36.6.1'); print(len(ex), 'examples'); print({e.get('source'):0 for e in ex})"
```

## Paths

| Role | Path |
|--|--|
| Snapshot | `data/hsreplay/36.6.1/` |
| Clean thin150 | `data/train_perfect_thin150_clean/` |
| Synth v2 | `data/train_synth_comps_v2.json` |
| Audit | `data/TRAIN_META_AUDIT.md` |
| Checkpoint | `results/eval_net_hsreplay_36_6_1.pt` → `ml/eval_net.pt` |

## Locked policy

- Comps highest weight; heroes/trinkets/spells support
- T7 useful but NOT required
- Naga out; no Firestone; no VODs

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
