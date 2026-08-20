"""Wire the LLM Turn Director into the live game loop.

`LiveCoach` (live.py) already turns the Power.log into snapshots + engine
advice at 20Hz. This module adds the Director layer on top WITHOUT touching
the fast path (spec req 6): the panel always renders instantly from the
engine, and a background worker asks the LLM for the one-move-at-a-time
line, swapping it in when it lands (~1-2s). If the LLM is down, slow, or
rejected by the validator, the engine's top move stands — the coach never
shows nothing.

Lifecycle handled here (spec reqs 5, 7, 13):
  * new game detected  -> build the GamePlan from the meta pack (Tier7-first
    when a snapshot exists) and the hero's strongest tribes.
    # CALIBRATE: the lobby's available-tribes line isn't parsed from the log
    yet, so the plan is seeded from the hero context + meta pack tribe
    ranking; once tribes are parseable the same call gets the real list.
  * game over          -> run the between-game reviewer in a background
    thread (grades our suggestions, appends lessons, promotes experiments,
    grows the training corpus), then reload lessons for the next game.
  * patch drift        -> `check_patch_drift` runs once at startup; the live
    build number isn't extracted from Power.log yet (# CALIBRATE), so drift
    warnings currently fire only when the pack itself is stale-dated.

Usage: created by the CLI (`watch --director`); render loops call
`augment(snap, lines)` which is non-blocking and returns the lines with the
Director's line on top.
"""

import os
import threading
import time
from typing import List, Optional, Tuple

from . import config
from .director import Suggestion, format_overlay_line, log_suggestion, suggest_move
from .game_plan import build_game_plan, plan_to_prompt, revise_plan, save_plan
from .llm_client import LLMClient, LLMError
from .meta_pack import check_patch_drift, load_meta_pack, pack_for_prompt
from .reviewer import (append_lessons, append_training_examples,
                       load_lessons_for_prompt, promote_experiments,
                       review_latest)


class DirectorLoop:
    """Background LLM suggestion worker + game lifecycle manager."""

    def __init__(self, client: Optional[LLMClient] = None,
                 data_dir: str = config.DATA_DIR, kb=None,
                 hero_ctx_fn=None, poll_s: float = 0.25):
        self.data_dir = data_dir
        self.kb = kb
        self._hero_ctx_fn = hero_ctx_fn or (lambda: None)
        self.poll_s = poll_s
        self.client = client
        self.client_error: Optional[str] = None
        if self.client is None:
            try:
                self.client = LLMClient()
            except LLMError as exc:            # misconfigured env -> engine-only
                self.client_error = str(exc)
        try:
            self.meta_pack = load_meta_pack()
        except (OSError, ValueError):
            self.meta_pack = None              # run `refresh-meta` to create it
        self.drift = check_patch_drift(self.meta_pack, None) if self.meta_pack else None
        self.lessons = load_lessons_for_prompt(
            os.path.join(data_dir, "lessons.jsonl"))
        self.plan = None
        self._plan_for_hero = None
        self._kb_names = ({c.name for c in kb.values()} if kb else None)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending: Optional[Tuple[dict, list, Optional[str]]] = None
        self._pending_key = None
        self._done_key = None
        self._suggestion: Optional[Suggestion] = None
        self._reviewing = False
        self._last_phase = None

    # -- render-loop side (non-blocking) ------------------------------------

    def augment(self, snap: dict, lines: List[str]) -> List[str]:
        """Return `lines` with the Director's line on top. Never blocks: on a
        state the worker hasn't finished yet, the engine's top line is shown
        as the interim pick."""
        phase = snap.get("phase")
        self._lifecycle(snap, phase)
        if phase not in ("recruit",) and not str(phase or "").startswith("choose"):
            return lines
        key = self._key(snap)
        with self._lock:
            self._pending = (snap, list(lines), self._choice_kind(phase))
            self._pending_key = key
            sug, done_key = self._suggestion, self._done_key
        out = list(lines)
        if self.meta_pack is None:
            out.insert(0, "DIRECTOR off — run `python -m hsbg_coach refresh-meta` first")
        elif self.client_error:
            out.insert(0, f"DIRECTOR (engine-only): {self.client_error[:70]}")
        elif sug is not None and done_key == key:
            out.insert(0, "DIRECTOR: " + format_overlay_line(sug))
        elif lines:
            out.insert(0, "DIRECTOR (thinking…): " + lines[0])
        if self.drift:
            out.append("⚠ " + self.drift)
        return out

    # -- lifecycle -----------------------------------------------------------

    def _lifecycle(self, snap: dict, phase) -> None:
        hero = snap.get("hero_name") or snap.get("hero")
        if hero and hero != self._plan_for_hero and self.meta_pack is not None:
            try:
                ctx = self._hero_ctx_fn()
                tribes = list(getattr(ctx, "best_tribes", None) or []) or \
                    [t.get("tribe") for t in self.meta_pack.get("tribes", [])[:4]
                     if t.get("tribe")]
                self.plan = build_game_plan(tribes, self.meta_pack)
                save_plan(self.plan, os.path.join(self.data_dir, "game_plan.json"))
                self._plan_for_hero = hero
            except Exception:
                pass
        if phase == "game_over" and self._last_phase != "game_over":
            self._spawn_review()
        self._last_phase = phase

    def _spawn_review(self) -> None:
        if self._reviewing:
            return
        self._reviewing = True

        def run():
            try:
                review = review_latest(self.data_dir, client=self.client)
                if review is not None:
                    append_lessons(review, os.path.join(self.data_dir, "lessons.jsonl"))
                    promote_experiments(review)
                    append_training_examples(review, review.game_path,
                                             os.path.join(self.data_dir,
                                                          "train_corpus.jsonl"))
                    self.lessons = load_lessons_for_prompt(
                        os.path.join(self.data_dir, "lessons.jsonl"))
            except Exception:
                pass                            # review must never hurt the live loop
            finally:
                self._reviewing = False

        threading.Thread(target=run, daemon=True).start()

    # -- worker side ---------------------------------------------------------

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self._work, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()

    def _work(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                job, key = self._pending, self._pending_key
                stale = (key == self._done_key)
            if job is None or stale or self.client is None or self.meta_pack is None:
                time.sleep(self.poll_s)
                continue
            snap, lines, choice_kind = job
            try:
                sug = self._suggest(snap, choice_kind)
            except Exception:
                sug = None
            with self._lock:
                if key == self._pending_key:    # board didn't move on — publish
                    self._suggestion = sug
                    self._done_key = key

    def _suggest(self, snap: dict, choice_kind: Optional[str]) -> Optional[Suggestion]:
        from .game_value import rank_actions
        snap = self._with_hero_power_text(snap)
        hero_ctx = self._hero_ctx_fn()
        try:
            recs, _base = rank_actions(snap, kb=self.kb, hero_ctx=hero_ctx)
        except Exception:
            recs = []
        lobby = (self.plan.lobby_tribes if self.plan else
                 [t.get("tribe") for t in (self.meta_pack or {}).get("tribes", [])[:4]
                  if t.get("tribe")])
        meta_ctx = pack_for_prompt(self.meta_pack, lobby) if self.meta_pack else ""
        plan_ctx = plan_to_prompt(self.plan) if self.plan else ""
        sug = suggest_move(snap, recs, meta_ctx, plan_ctx, self.lessons,
                           self.client, kb_names=self._kb_names,
                           choice_kind=choice_kind)
        if sug.plan_update and self.plan is not None:
            self.plan = revise_plan(self.plan, sug.plan_update)
            save_plan(self.plan, os.path.join(self.data_dir, "game_plan.json"))
        try:
            log_suggestion(sug, snap, recs,
                           dir=os.path.join(self.data_dir, "suggestions"))
        except Exception:
            pass
        return sug

    def _with_hero_power_text(self, snap: dict) -> dict:
        """Attach the hero power's real effect text (from the committed
        hero-power KB) so the LLM reasons from what the power actually does,
        never from memory (spec req 8)."""
        hp = snap.get("hero_power")
        if not isinstance(hp, dict) or hp.get("text"):
            return snap
        if not hasattr(self, "_hp_kb"):
            try:
                from .cards import load_hero_powers
                self._hp_kb = load_hero_powers()
            except Exception:
                self._hp_kb = {}
        row = (self._hp_kb.get(hp.get("card_id") or "")
               or self._hp_kb.get((hp.get("name") or "").lower()))
        if row and row.get("text"):
            return dict(snap, hero_power=dict(hp, text=row["text"]))
        return snap

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _choice_kind(phase) -> Optional[str]:
        p = str(phase or "")
        if p.startswith("choose "):
            kind = p.split(" ", 1)[1]
            return {"hero": "hero", "trinket": "trinket", "discover": "discover",
                    "dark_gift": "dark_gift"}.get(kind, "discover")
        return None

    @staticmethod
    def _key(snap: dict):
        board = tuple((m.get("name"), m.get("attack"), m.get("health"))
                      for m in snap.get("board", []) or [])
        shop = tuple((m.get("name"), m.get("attack"), m.get("health"))
                     for m in snap.get("shop", []) or [])
        return (snap.get("turn"), snap.get("gold"), snap.get("tavern_tier"),
                snap.get("phase"), board, shop)
