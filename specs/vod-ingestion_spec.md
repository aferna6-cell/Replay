# Spec: Streamer VOD ingestion (transcripts → vision)

**Goal.** Get high-level human gameplay into the training loop. Our own logs
capture *our* decisions; population stats capture *what* wins on average; VODs
of top players are the only accessible source of *why* strong players make the
decisions they make, and (via the vision rung) of raw high-MMR trajectories we
can't otherwise obtain (HSReplay has no public API; nobody publishes raw BG
trajectory dumps — see README "Data sourcing").

## Rung 1 — transcripts (BUILT: `scripts/fetch_vod_transcript.py`)

What it is: `yt-dlp`-based fetcher for YouTube/Twitch VOD captions →
`data/vods/<id>.txt` ([mm:ss]-stamped, deduped) + `<id>.info.json` metadata.
Local-only (`data/vods/` is gitignored — other people's content stays off the
repo).

What transcripts are good for — and not:

| Signal | In transcripts? |
|---|---|
| Strategic reasoning ("greeding for tier 5 here because…") | yes — the unique value |
| Timing conventions (level curves, when to freeze) | partially, when narrated |
| Exact board/shop states, gold, hp | **no** — needs vision (rung 2) |
| (state, action, outcome) training tuples | **no** — needs vision (rung 2) |

Consumption (next step, not yet built): distill transcripts with an LLM into
structured "decision rules with conditions" (e.g. curve conventions per hero
archetype, pivot triggers), reviewed by hand, then merged into the advisor's
heuristic layer (`advisor.py` / `build_path.py` weights) — NOT fed raw to the
eval net. Transcripts are commentary, not ground truth; they tune priors,
they don't label data.

## Rung 2 — vision state reconstruction (NOT BUILT; the big one)

Reconstruct turn-by-turn `(state, action, outcome)` trajectories from video:

1. **Frame sampling** — `yt-dlp` download → `ffmpeg` keyframes ~1 fps.
2. **Phase segmentation** — recruit vs combat via cheap template cues (Bob's
   board banner, combat animation): OpenCV template match, no ML needed.
3. **State reading (hard part)** — per recruit-phase frame, read board + shop
   minions, gold, tier, hp, turn. Two candidate approaches, decide by trial:
   a. Vision LLM per sampled frame (costly but zero training; card art +
      overlaid stat digits are exactly what VLMs read well now).
   b. Classic CV: card-art embedding match against `data/cards/` art + digit
      OCR at fixed HUD offsets (cheap at scale, brittle to skins/patches).
4. **Action inference** — diff consecutive states (bought X, sold Y, rolled,
   leveled) — same event grammar the Power.log recorder emits, so
   reconstructed trajectories drop straight into `data/*.jsonl` and train the
   eval net with `--trajectories` unchanged.
5. **Outcome** — final placement read from the endgame screen.

Open questions (resolve before building):
- Per-VOD cost of (a) at ~1 frame/3s over a 4-hour VOD — likely needs combat
  skipping + recruit-only sampling to be sane.
- Streamer overlay occlusion (webcams over the shop) — may need per-streamer
  crop configs.
- Accuracy gate: validate reconstructed trajectories against a VOD of OUR OWN
  game where the Power.log ground truth exists; ship only if state accuracy
  >95% on board contents / gold / tier.

## Non-goals

- Live stream parsing (VODs only).
- Committing transcripts or video to the repo.
- Training the eval net directly on transcript text.
