"""Claude-vision frame reading: phase classification + recruit-state extraction.

Two tiers, deliberately different models:
  * classify_frame — "recruit / combat / endscreen / other" on a downscaled
    thumbnail. Trivial classification -> Claude Haiku (cheap, called on every
    deduped frame).
  * read_state — full structured read of a recruit/endscreen frame ->
    Claude Opus 5 (needs to read card names + small HUD digits; called only on
    the frames that matter). Structured output (json_schema) so parsing never
    drifts.

Credentials resolve like any Anthropic SDK app (ANTHROPIC_API_KEY or an
`ant auth login` profile). Opus 5 requests carry the server-side refusal
fallback by default; if the API rejects those beta params (older gateway),
we retry once without them and remember.
"""

import base64
import io
import json
from pathlib import Path
from typing import Dict, Optional

from PIL import Image

READ_MODEL = "claude-opus-5"
CLASSIFY_MODEL = "claude-haiku-4-5"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Long-edge caps: HUD digits stay readable at 1600px; thumbnails classify fine.
_READ_EDGE = 1600
_CLASSIFY_EDGE = 512

PHASES = ("recruit", "combat", "endscreen", "other")

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {"phase": {"type": "string", "enum": list(PHASES)}},
    "required": ["phase"],
    "additionalProperties": False,
}

_MINION = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "attack": {"type": ["integer", "null"]},
        "health": {"type": ["integer", "null"]},
        "golden": {"type": "boolean"},
    },
    "required": ["name", "attack", "health", "golden"],
    "additionalProperties": False,
}

STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "phase": {"type": "string", "enum": list(PHASES)},
        "turn": {"type": ["integer", "null"],
                 "description": "turn counter if visible, else null"},
        "tavern_tier": {"type": ["integer", "null"]},
        "gold": {"type": ["integer", "null"]},
        "hero_health": {"type": ["integer", "null"]},
        "hero_name": {"type": ["string", "null"]},
        "board": {"type": "array", "items": _MINION,
                  "description": "the player's warband, left to right"},
        "shop": {"type": "array", "items": _MINION,
                 "description": "Bob's shop minions, left to right"},
        "final_placement": {"type": ["integer", "null"],
                            "description": "1-8 on an endscreen, else null"},
        "confidence": {"type": "number",
                       "description": "0-1: how readable was this frame "
                                      "(occlusion, motion blur, overlays)"},
    },
    "required": ["phase", "turn", "tavern_tier", "gold", "hero_health",
                 "hero_name", "board", "shop", "final_placement", "confidence"],
    "additionalProperties": False,
}

_READ_PROMPT = """\
This is a frame from a Hearthstone Battlegrounds VOD. Read the visible game
state exactly — do not guess values that are occluded (webcam, overlays) or
unreadable; use null for those and lower `confidence`.

- `board`: the player's own warband (bottom row in recruit phase).
- `shop`: Bob's offered minions (top row in recruit phase).
- `gold`: current gold (coin counter). `tavern_tier`: the tavern level.
- `turn`: the turn number if the UI or a tracker overlay shows it, else null.
- `hero_health`: the player's hero health.
- On a results/endscreen, set `final_placement` to the numeric place shown.
- Use each card's exact English card name; mark goldens with `golden: true`."""


class FrameReader:
    def __init__(self, read_model: str = READ_MODEL,
                 classify_model: str = CLASSIFY_MODEL):
        import anthropic  # deferred: only the ingest path needs the SDK
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.read_model = read_model
        self.classify_model = classify_model
        self._use_fallbacks = read_model.startswith("claude-opus-5")
        self.calls = {"classify": 0, "read": 0}

    # -- image plumbing ----------------------------------------------------
    @staticmethod
    def _b64(path: Path, long_edge: int) -> str:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = long_edge / max(w, h)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=87)
        return base64.standard_b64encode(buf.getvalue()).decode()

    def _image_block(self, path: Path, long_edge: int) -> Dict:
        return {"type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": self._b64(path, long_edge)}}

    # -- API calls ---------------------------------------------------------
    def _structured(self, model: str, blocks, schema, max_tokens: int) -> Optional[Dict]:
        kwargs = dict(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": blocks}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        try:
            if self._use_fallbacks and model == self.read_model:
                resp = self.client.beta.messages.create(
                    betas=[_FALLBACK_BETA], fallbacks="default", **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except self._anthropic.BadRequestError as exc:
            if self._use_fallbacks and ("fallback" in str(exc).lower()
                                        or "beta" in str(exc).lower()):
                self._use_fallbacks = False   # older gateway: retry plainly
                return self._structured(model, blocks, schema, max_tokens)
            raise
        if resp.stop_reason == "refusal":
            return None
        text = next((b.text for b in resp.content
                     if getattr(b, "type", "") == "text"), None)
        return json.loads(text) if text else None

    def classify_frame(self, path: Path) -> str:
        self.calls["classify"] += 1
        out = self._structured(
            self.classify_model,
            [self._image_block(path, _CLASSIFY_EDGE),
             {"type": "text", "text":
              "One frame from a Hearthstone Battlegrounds video. Classify it: "
              "'recruit' (Bob's Tavern shop visible), 'combat' (two warbands "
              "fighting), 'endscreen' (final placement/results), or 'other' "
              "(menus, hero pick, loading, non-game content)."}],
            CLASSIFY_SCHEMA, max_tokens=64)
        return out["phase"] if out and out.get("phase") in PHASES else "other"

    def read_state(self, path: Path) -> Optional[Dict]:
        self.calls["read"] += 1
        return self._structured(
            self.read_model,
            [self._image_block(path, _READ_EDGE),
             {"type": "text", "text": _READ_PROMPT}],
            STATE_SCHEMA, max_tokens=2048)
