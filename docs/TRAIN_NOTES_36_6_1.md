# TRAIN_NOTES — 36.6.1 Aberrations (2026-09-22)

**Goal:** Ship an `hsbg_coach` eval-net checkpoint trained on today's patch VODs with denser **gift > body**, buy/sell/freeze/HP/level/roll, and tribe-direction signals. Naga forever excluded. `wigwPzucfXc` skipped (mislabeled pre-patch).

## Checkpoint (primary)

| | |
|--|--|
| **Path (repo)** | `results/eval_net_36_6_1_2026-09-22/eval_net.pt` |
| **Live default** | `ml/eval_net.pt` (copy of primary) |
| **Pilot mirror** | `/workspace/hsreplay-tier7/jeef_vod_pilot/train_patch_36_6_1_today/eval_net.pt` |
| **Alt (VOD-only)** | `results/eval_net_36_6_1_2026-09-22/eval_net_vod_only.pt` |

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
| eval_net_vod_only.pt | 40 | 0.266 | 0.493 |

Population base still ~11.5k HSReplay comps. VOD boards carry `weight_hint=4` in JSON (trainer currently unweighted by hint; featured mix still isolates the scrubbed corpus).

## Corpus

| Metric | Count |
|--------|------:|
| Train JSONL rows (`data/expert/expert_aberration_vods.jsonl`) | **416** |
| Decisions | **227** |
| Gift-related decisions | **19** |
| Buy decisions | **23** |
| Sell / roll / freeze / HP / level | 36 / 35 / 8 / 9 / 14 |
| Placeable boards (placement + ≥2 minions) | **40** |
| Skipped `wigwPzucfXc` rows | 139 |
| Scrubbed Naga rows | 14 |

Sources: Jeef shorts + Shadybunny patch-day + prior RDU Day1/Day2 EA (non-Naga). Soft priors in `hsbg_coach/jeef_priors.py` already encode gift>body / gift+HP / Activate; denser ASR rows reinforce those plans (`dark_gift_over_level`, `dark_gift_chef_hp`, etc.).

### URLs ingested this hunt

| URL | Role | Status |
|-----|------|--------|
| https://youtu.be/z9PfvDgMqAU | Jeef short (label pilot) | labeled + ASR densified |
| https://youtu.be/g8kCF5Z-UIs | Jeef short | labeled + ASR densified |
| https://youtu.be/V0ZyTYC9gr8 | Jeef spot-check | labeled + ASR densified |
| https://youtu.be/pzYwWAnjJ54 | Shadybunny ~5h patch-day | 720p downloaded; **123 ASR decision rows** |
| https://youtu.be/0Srr-tgf5Vo | RDU Day2 (prior) | kept (non-Naga lobbies) |
| https://youtu.be/Bpr_xTUjktA | RDU Day1 (prior) | kept (Naga lobby already Q) |
| https://www.twitch.tv/videos/2880966040 | RDU Twitch ~8.5h | **partial ~9%** at train time (yt-dlp running; not in this checkpoint) |
| https://youtu.be/wigwPzucfXc | JeefHSVODs | **SKIPPED** |

Optional XQN / Dogdog Twitch VODs not fetched this pass (disk/time).

## Pipeline reused

- Labels: `/workspace/hsreplay-tier7/jeef_vod_pilot/labels/` (+ densified Jeef shorts / Shadybunny)
- Densify: `scripts/vod_pilot/densify_asr_decisions.py`
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

1. Finish RDU Twitch `2880966040` download → lobby detect → densify → retrain.
2. Optional XQN / Dogdog VODs.
3. Vision pass for cardId+tier+tribe on Shadybunny gift/buy peaks (ASR-only needs_review today).
4. When HSReplay Aberration perfect games appear, upweight those (`weight_hint=3`) over VOD.
