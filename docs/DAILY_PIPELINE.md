# Daily Firestone pipeline

One command pulls every Firestone game we don't have yet, builds `states.v2`, labels it
(`labels.v2`), retrains the behaviour-cloned NEXT policy on the whole labeled corpus,
and installs the new policy only if it earns it. Code: `hsbg_coach/daily_pipeline.py`
(orchestration) and `hsbg_coach/firestone_replays.py` (fetch).

## Run it locally (from the Replay folder)

```powershell
cd C:\Users\aidan\Replay
git fetch origin; git checkout claude/nifty-archimedes-88wgst    # until the PR merges
.\scripts\daily_firestone_pipeline.ps1                           # fetch + label + train + gated install
```

That is the whole job, with output on screen. Common variants (`-PipelineArgs` takes
any flag listed below):

```powershell
.\scripts\daily_firestone_pipeline.ps1 -PipelineArgs '--force-train --max-new 0'  # just retrain on what you have
.\scripts\daily_firestone_pipeline.ps1 -PipelineArgs '--no-install'              # train + report, keep the live policy
.\scripts\daily_firestone_pipeline.ps1 -PipelineArgs '--no-train'                # data only
.\.venv\Scripts\python.exe -m hsbg_coach.daily_pipeline --status                   # corpus + last runs
```

Needs: the `.venv` you already train with (NumPy + Torch), and `ml\eval_net.pt`
(the gate's advisor baseline; without it nothing is ever installed). Your corpus
manifest must be `data\firestone\raw\manifest.json`, or add
`-PipelineArgs '--manifest data\firestone\raw\<yours>.json'`; the job stops with that
hint if it sees replays in `raw\` but no manifest. The first run also downloads
everything Firestone currently lists that you don't have (up to ~1,000 games), so it
is the long one. Each run's details: `data\firestone\batches\<id>\status.json` and
`train.log`; the candidate + metrics are in `results\policy_net_<id>.*`.

## Set it up once (Windows)

```powershell
cd C:\Users\aidan\Replay
.\scripts\install_daily_pipeline_task.ps1            # every day at 04:30
Start-ScheduledTask -TaskName 'Replay daily Firestone pipeline'   # run it now
.\.venv\Scripts\python.exe -m hsbg_coach.daily_pipeline --status  # corpus + last runs
```

Logs: `data\firestone\logs\daily-YYYY-MM-DD.log`. mac/linux: cron
`30 4 * * * /path/to/Replay/scripts/daily_firestone_pipeline.sh`.

## What one run does

| Stage | What | Where |
|---|---|---|
| fetch | Firestone's public list of recent first-place games (`static.zerotoheroes.com/api/bgs/bgs-perfect-games.json`, the latest ~1,000, about 4 days). Skips every `reviewId`/`originalReviewId` already in the manifest or `raw/`. Each replay (`xml.firestoneapp.com/<replayKey>`, a zip) becomes `<reviewId>.xml.gz`, with a manifest entry in the shape `replay_states` / `replay_labels` read | `batches/<id>/raw` |
| states | `replay_states.run` on the batch only, split over `--workers` processes | `batches/<id>/states` |
| merge | replays, states, quarantine into the corpus; entries appended to the manifest | `raw/`, `states/v2/` |
| labels | `replay_labels.run` on the batch, plus (once each) any corpus game that has states but no labels (e.g. games built before this job); MMR weights from the whole corpus manifest | `batches/<id>/labels` |
| merge | labels + quarantine into the corpus | `labels/v2/` |
| train | `python -m ml.train_bc_policy --data labels/v2/*.jsonl.gz --make-heldout --compare ml/policy_net.pt` (no `--install`) | `results/policy_net_<id>.pt` + `.metrics.json`, `batches/<id>/train.log` |
| promote | install only if the gate is **PASS**, held-out top-1 is not below the live policy (`--min-gain`, default 0), and `scripts/policy_smoke.py` passes on the candidate; after install, a failed live smoke rolls back to `policy_net.prev.pt` | `ml/policy_net.pt` |

`train_bc_policy --install` installs even when its gate fails, so the daily job
never passes it; it installs through `train_bc_policy.install()` after its own checks.
The gate needs the eval-net advisor (`ml/eval_net.pt`); without it nothing is installed.

The held-out set is created on the first train (`results/policy_heldout_ids.json`)
and then stays frozen, so every day's candidate is scored on the same games.

## Failure behaviour

- Every stage writes to the batch folder first, so a stage never re-processes, and
  never quarantines, games from earlier days.
- A crash in a data stage leaves the batch unfinished; the next run resumes it at that
  stage, then runs that day's batch.
- A failed train/promote is recorded and retried by the next run
  (`pipeline_state.json: train_pending`), even if no new games came in.
- Network hiccups (resets, timeouts, a truncated list download) are retried.
- A failed replay download is not recorded anywhere, so the next run tries it again
  while Firestone still lists it.
- One run at a time (`data/firestone/.daily_pipeline.lock`, stale after 20 h).

## Useful flags

`--no-train` (data only) · `--no-install` (train + report) · `--force-train` ·
`--max-new N` · `--workers N` · `--no-opponents` (states ~10x faster, no opponent
fields) · `--build B` · `--train-arg=--epochs=20` (any trainer flag) ·
`--manifest PATH` (if your corpus manifest is not `data/firestone/raw/manifest.json`).

## Limits

- Firestone's list covers only the last ~4 days; games from days the PC was off
  longer than that are gone.
- Only first-place games are listed, which is what `labels.v2` trains on anyway.
- Label MMR weights are fixed when a batch is labeled; they are percentiles of the
  corpus at that time, so older batches drift slightly as the corpus grows.
- Training runs over the whole corpus every day; on CPU that is the slow stage.
