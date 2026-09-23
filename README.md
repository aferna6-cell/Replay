# hsbg-coach

Live **Hearthstone Battlegrounds** coach: tail `Power.log`, reconstruct BG state, show a NEXT overlay, and score boards with a local eval net trained from an HSReplay snapshot (patch **36.6.1**).

The Power.log logger is still the foundation. The eval net on top is what drives live board advice.

```
Power.log -> parse -> BG state -> overlay NEXT <- eval_net.pt (HSReplay snapshot train)
```

## Status (2026-09-23)

| Layer | State |
|---|---|
| Power.log detect / setup / tail / parse | working on Windows |
| BG state + live overlay | working; restart overlay after retrain |
| Eval net (`ml/eval_net.pt`) | trained on scrubbed 36.6.1 HSReplay corpus |
| Train corpus | 11 intact comps + 5 synth (all 10 live tribes) + 72 clean thin150 + heroes/trinkets/spells support |
| Out of this train | Firestone live fetch, VOD corpus, naga, out-of-pool keys |

**Ready for in-game playtest** after you retrain with the current support weights (comps 4×, heroes/trinkets/spells support 2×).

Latest scrubbed-corpus metrics (pre-2×-support bump): val MAE **0.617**, Pearson **r 0.805**. Re-run train below so the checkpoint matches the 2× support bump before judging overlay quality.

## Quick start (play)

```powershell
cd C:\Users\aidan\Replay
.".venv\Scripts\python.exe" -m hsbg_coach detect
.".venv\Scripts\python.exe" -m hsbg_coach setup   # then restart Hearthstone
.".venv\Scripts\python.exe" -m hsbg_coach watch --overlay
```

Use borderless/windowed Hearthstone. After any retrain, **restart the overlay** so it reloads `ml/eval_net.pt`.

## Retrain eval net (local, HSReplay-only)

Canonical PowerShell (from repo root):

```powershell
cd C:\Users\aidan\Replay
New-Item -ItemType Directory -Force -Path results | Out-Null
.\.".venv\Scripts\python.exe" -m ml.train_eval_net `
  --hsreplay-snapshot data/hsreplay/36.6.1 `
  --trajectories data/train_perfect_thin150_clean `
  --epochs 40 `
  --out results/eval_net_hsreplay_36_6_1.pt
Copy-Item results\eval_net_hsreplay_36_6_1.pt ml\eval_net.pt -Force
```

Same commands live in `data/TRAIN_LOCAL_COMMANDS.md` and `docs/TRAIN_HSREPLAY_LOCAL.md`.

### Train policy (locked)

- **Comps highest** (4× copies). Heroes / trinkets / spells are **support** (2×).
- Only comps whose **every** core / key / enabler / addon is still in the live minion pool.
- Missing tribes filled with **approved synth comps v2** (playbooks + real synergies; **no T7 required** — T7 is useful, not mandatory).
- Perfect boards: **clean thin150 only** (`data/train_perfect_thin150_clean/`); dirty/out-of-pool boards excluded.
- **Naga out.** No Firestone live fetch. No VOD corpus for this train.
- Meta notes in snapshot: aberrations strong **early–mid then pivot**; quilboar often **support**; discard is a legal action.

Audit: `data/TRAIN_META_AUDIT.md`. Synth pack: `data/train_synth_comps_v2.json`.

### Snapshot layout

`data/hsreplay/36.6.1/` — `comps.json`, `minions.json`, `heroes.json`, `trinkets.json`, `spells.json`, `tribes.json`, `meta.json`.

## Data sourcing (design)

| Source | Role |
|---|---|
| HSReplay snapshot (comps / heroes / trinkets / spells / minions) | Population prior for eval-net train |
| Clean perfect thin150 boards | Trajectory examples folded into train |
| Your own games (this logger) | Personalization later; not required for the current eval-net train |

Population vs personal weighting stays adaptive long-term (`WEIGHTING` in `hsbg_coach/config.py`). The current eval checkpoint is **population / snapshot only**.

## Logger / parser (still core)

```powershell
# Offline: parse a previously captured log
.\.".venv\Scripts\python.exe" -m hsbg_coach parse-file path\to\Power.log

# Overlay with sample data only
.\.".venv\Scripts\python.exe" -m hsbg_coach overlay

# Terminal recommendation panel instead of overlay
.\.".venv\Scripts\python.exe" -m hsbg_coach watch --terminal
```

| Layer | State |
|---|---|
| Log path detection (Mac + Windows) | implemented |
| `log.config` writer | implemented |
| Log tailer | implemented |
| Raw line parser | implemented + unit-tested |
| BG semantic layer | calibrated; shop zone + placement still want real-log confirmation |
| `(state, action, outcome)` recorder | implemented |
| Monte Carlo combat sim | implemented + unit-tested |
| Overlay UI | implemented |

See `specs/hsbg-coach_spec.md` for calibration notes.

## Playtest checklist

1. Retrain with the commands above (picks up support 2×).
2. `Copy-Item` into `ml/eval_net.pt` and restart overlay.
3. In lobby, watch: early aberration buys / mid pivot; hero·trinket·spell NEXT; no out-of-pool junk.
4. If a tribe feels wrong, note lobby + NEXT text — reweight or rewrite that synth, then retrain.

## Docs

- `docs/TRAIN_HSREPLAY_LOCAL.md` — local HSReplay train
- `data/TRAIN_LOCAL_COMMANDS.md` — copy-paste PowerShell
- `data/TRAIN_META_AUDIT.md` — pool / comps / thin150 audit
- `data/TRAIN_REVIEW_PROPOSED.md` — kept + synth boards for review
