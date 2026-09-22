# TRAIN_NOTES — 36.6.1 Aberrations (2026-09-22)

**Goal:** Ship an `hsbg_coach` eval-net checkpoint trained on today's patch VODs with denser **gift > body**, buy/sell/freeze/HP/level/roll, and tribe-direction signals. Naga forever excluded. `wigwPzucfXc` skipped (mislabeled pre-patch).

## Checkpoint (primary)

| | |
|--|--|
| **Path (repo)** | `results/eval_net_36_6_1_2026-09-22/eval_net.pt` |
| **Live default** | `ml/eval_net.pt` (copy of primary) |
| **Pilot mirror** | `/workspace/hsreplay-tier7/jeef_vod_pilot/train_patch_36_6_1_today/eval_net.pt` |
| **Cycle** | **3** — full RDU Twitch `2880966040` Whisper ASR densify (2026-09-22 ~7:00 PM ET) |

### Point `watch --overlay` at the new weights

```bash
cd /tmp/Replay-hsreplay
cp results/eval_net_36_6_1_2026-09-22/eval_net.pt ml/eval_net.pt
python -m hsbg_coach watch --overlay
```

## Train metrics (40 ep, CPU)

| Checkpoint | Recorded boards | val MAE | val Pearson r |
|------------|----------------:|--------:|--------------:|
| **eval_net.pt (VOD + thin150)** | 190 (40 VOD placeable + 150 thin) | **0.337** | **0.560** |

Placeable boards unchanged across cycles 1–3 (ASR densifies decision coverage; eval-net trains on placement boards). Population base ~11.5k HSReplay comps.

## Corpus (cycle 3)

| Metric | #86 | #87 | #89 | **Now (#cycle3)** |
|--------|----:|----:|----:|------------------:|
| Train JSONL rows | 416 | 499 | 559 | **618** |
| Decisions | 227 | 310 | 370 | **429** |
| Gift-related decisions | 19 | 32 | 36 | **48** |
| Buy decisions | 23 | 47 | 83 | **96** |
| Sell / roll / freeze / HP / level | 36/35/8/9/14 | — | 60/45/13/16/16 | **63/52/18/26/20** |
| Placeable boards | 40 | 40 | 40 | **40** |
| Skipped `wigwPzucfXc` | 139 | 139 | 139 | 139 |
| Scrubbed Naga | 14 | 14 | 14 | 14 |

Sources: Jeef shorts + Shadybunny + RDU Day1/Day2 EA + **RDU Twitch full ~8.5h Whisper**. Soft priors in `hsbg_coach/jeef_priors.py` already encode gift>body / gift+HP / Activate.

### URLs

| URL | Role | Status |
|-----|------|--------|
| Jeef shorts `z9PfvDgMqAU` / `g8kCF5Z-UIs` / `V0ZyTYC9gr8` | EA / live | densified |
| https://youtu.be/pzYwWAnjJ54 | Shadybunny ~5h | densified (145+) |
| https://youtu.be/0Srr-tgf5Vo / `Bpr_xTUjktA` | RDU Day2/Day1 | densified |
| https://youtu.be/Ry-Zn2sPP2k / `1BZtAVX8n50` | RDU EA | densified |
| https://www.twitch.tv/videos/2880966040 | RDU Twitch ~8.5h | **COMPLETE** Whisper densify (~78 decision rows) |
| https://youtu.be/wigwPzucfXc | JeefHSVODs | **SKIPPED** |
| XQN `2881206250` / Dogdog `2881206616` | optional | deferred (XQN VOD still live-tailing / incomplete index) |

## Pipeline

- Densify: `scripts/vod_pilot/densify_asr_decisions.py`
- Merge: `scripts/vod_pilot/merge_aberration_vod_corpus.py` (drops `wigwPzucfXc` + Naga)
- Train:
```bash
.venv/bin/python -m ml.train_eval_net --epochs 40 \
  --trajectories data_vod_patch_36_6_1_today/ \
  --out results/eval_net_36_6_1_2026-09-22/eval_net.pt
cp results/eval_net_36_6_1_2026-09-22/eval_net.pt ml/eval_net.pt
```

## Naga policy

Forever exclude. Quarantine: `jeef_vod_pilot/labels/naga_quarantine.jsonl`.

## Next

1. Vision/lobby pass on full Twitch VOD for placeable boards (MAE lever).
2. Retry XQN/Dogdog when VODs finalize.
3. HSReplay Aberration perfect games → `weight_hint=3`.
