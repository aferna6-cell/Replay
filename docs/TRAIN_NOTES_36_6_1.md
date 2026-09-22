# TRAIN_NOTES — 36.6.1 Aberrations (2026-09-22)

**Goal:** Ship an `hsbg_coach` eval-net checkpoint trained on today's patch VODs with denser **gift > body**, buy/sell/freeze/HP/level/roll, and tribe-direction signals. Naga forever excluded. `wigwPzucfXc` skipped (mislabeled pre-patch).

## Checkpoint (primary)

| | |
|--|--|
| **Path (repo)** | `results/eval_net_36_6_1_2026-09-22/eval_net.pt` |
| **Live default** | `ml/eval_net.pt` (copy of primary) |
| **Pilot mirror** | `/workspace/hsreplay-tier7/jeef_vod_pilot/train_patch_36_6_1_today/eval_net.pt` |
| **Cycle** | **1b** — RDU YT ASR densify + fixed Shadybunny VTT parse (2026-09-22 ~6:25 PM ET) |

### Point `watch --overlay` at the new weights

```bash
cd /tmp/Replay-hsreplay   # or your Replay checkout
cp results/eval_net_36_6_1_2026-09-22/eval_net.pt ml/eval_net.pt
python -m hsbg_coach watch --overlay
```

`get_scorer()` / live coach load `ml/eval_net.pt` by default — no CLI flag required once the file is in place.

## Train metrics (40 ep, CPU, torch 2.14)

| Checkpoint | Recorded boards | val MAE | val Pearson r |
|------------|----------------:|--------:|--------------:|
| **eval_net.pt (VOD + thin150)** | 190 (40 VOD placeable + 150 thin) | **0.337** | **0.560** |
| eval_net_vod_only (prior) | 40 | 0.266 | 0.493 |

Placeable board count unchanged this cycle (ASR decisions densify gift/buy coverage; eval-net still trains on placement boards). Population base still ~11.5k HSReplay comps.

## Corpus (cycle 1b)

| Metric | Prior (#86) | **Now** |
|--------|------------:|--------:|
| Train JSONL rows | 416 | **499** |
| Decisions | 227 | **310** |
| Gift-related decisions | 19 | **32** |
| Buy decisions | 23 | **47** |
| Sell / roll / freeze / HP / level | 36/35/8/9/14 | **52/43/10/16/14** |
| Placeable boards (placement + ≥2 minions) | 40 | **40** |
| Skipped `wigwPzucfXc` rows | 139 | 139 |
| Scrubbed Naga rows | 14 | 14 |

Sources: Jeef shorts + Shadybunny patch-day + RDU Day1/Day2 EA (non-Naga). Cycle 1b densified ASR on `0Srr-tgf5Vo`, `Bpr_xTUjktA`, `Ry-Zn2sPP2k`, `1BZtAVX8n50`, and re-parsed Shadybunny VTT (`pzYwWAnjJ54`). Soft priors in `hsbg_coach/jeef_priors.py` already encode gift>body / gift+HP / Activate.

### URLs ingested this hunt

| URL | Role | Status |
|-----|------|--------|
| https://youtu.be/z9PfvDgMqAU | Jeef short (label pilot) | labeled + ASR densified |
| https://youtu.be/g8kCF5Z-UIs | Jeef short | labeled + ASR densified |
| https://youtu.be/V0ZyTYC9gr8 | Jeef spot-check | labeled + ASR densified |
| https://youtu.be/pzYwWAnjJ54 | Shadybunny ~5h patch-day | 720p; VTT densify (**145** rows) |
| https://youtu.be/0Srr-tgf5Vo | RDU Day2 | kept + **ASR densified** |
| https://youtu.be/Bpr_xTUjktA | RDU Day1 | kept + **ASR densified** (Naga lobby Q) |
| https://youtu.be/Ry-Zn2sPP2k | RDU EA | densified |
| https://youtu.be/1BZtAVX8n50 | RDU EA | densified |
| https://www.twitch.tv/videos/2880966040 | RDU Twitch ~8.5h | **downloading** (~38% at cycle 1b ship) |
| https://youtu.be/wigwPzucfXc | JeefHSVODs | **SKIPPED** |

Optional XQN `2881206250` / Dogdog `2881206616` deferred until RDU Twitch finishes (disk/bandwidth).

## Pipeline reused

- Labels: `/workspace/hsreplay-tier7/jeef_vod_pilot/labels/`
- Densify: `scripts/vod_pilot/densify_asr_decisions.py` (RDU+Shadybunny targets; fixed YouTube auto-VTT parse)
- Merge/scrub: `scripts/vod_pilot/merge_aberration_vod_corpus.py` (drops `wigwPzucfXc` + Naga)
- Train:
```bash
cd /tmp/Replay-hsreplay
.venv/bin/python -m ml.train_eval_net \
  --epochs 40 \
  --trajectories data_vod_patch_36_6_1_today/ \
  --out results/eval_net_36_6_1_2026-09-22/eval_net.pt
cp results/eval_net_36_6_1_2026-09-22/eval_net.pt ml/eval_net.pt
```

## Naga policy

Forever exclude. Quarantine file remains `jeef_vod_pilot/labels/naga_quarantine.jsonl`. Merge drops any row matching `\bnaga` / `exclude_from_train`.

## Next

1. Finish RDU Twitch `2880966040` → lobby detect → Whisper ASR densify → retrain (cycle 2).
2. Optional XQN / Dogdog VODs.
3. Vision pass for cardId+tier+tribe on gift/buy peaks (ASR-only needs_review today).
4. When HSReplay Aberration perfect games appear, upweight those (`weight_hint=3`) over VOD.
