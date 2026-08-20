"""Server-agnostic chat client for the Turn Director's open-weight LLM.

Phase 3 (`specs/llm-turn-director_spec.md` reqs 3/6) needs one move + a
one-line why out of a small, fast model, served from wherever the owner's
`llm-bench` numbers say is best: a local Ollama/llama.cpp instance on the
gaming PC, or a hosted open-weights endpoint (Groq/Together/OpenRouter/LM
Studio/llama.cpp server all speak the same OpenAI-compatible `/v1/chat/
completions` shape). This module hides that choice behind one `chat()` call
so `director.py` never has to know which server is on the other end.

Two backends:

  * ``ollama``  -> POST ``{url}/api/chat`` (non-streaming), ``format: "json"``
    when JSON output is requested.
  * ``openai``  -> POST ``{url}/v1/chat/completions``, ``response_format:
    {"type": "json_object"}`` when JSON output is requested. Works unchanged
    against Together/Groq/OpenRouter/LM Studio/llama.cpp's server mode — they
    all implement this same request/response shape.

Stdlib only (``urllib``), like the rest of this repo. The only network seam is
``LLMClient._post`` — tests monkeypatch it (or, for transport-error mapping,
``urllib.request.urlopen``) so nothing here ever touches a real network.
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional

OLLAMA = "ollama"
OPENAI = "openai"

DEFAULT_BACKEND = OLLAMA
DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 8
DEFAULT_MAX_TOKENS = 300


class LLMError(Exception):
    """The LLM server couldn't be reached, errored, or replied unusably.

    Always carries an actionable, human-readable message — the Director
    catches this and falls back to the engine's own top move, but the
    message is worth surfacing to the owner (e.g. in a `--verbose` log)."""


@dataclass
class LLMConfig:
    """Everything needed to talk to one LLM backend for one game session."""

    backend: str = DEFAULT_BACKEND               # "ollama" | "openai"
    model: str = DEFAULT_MODEL
    url: str = DEFAULT_OLLAMA_URL
    api_key: Optional[str] = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_tokens: int = DEFAULT_MAX_TOKENS

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Read HSBG_LLM_* env vars. Ollama gets a working default URL (the
        common local-dev case); any hosted/OpenAI-compatible backend must
        set HSBG_LLM_URL explicitly — there's no safe default to guess for a
        remote API endpoint, so we fail fast with an actionable message
        instead of silently pointing nowhere."""
        backend = (os.environ.get("HSBG_LLM_BACKEND") or DEFAULT_BACKEND).strip().lower()
        model = os.environ.get("HSBG_LLM_MODEL") or DEFAULT_MODEL
        url = os.environ.get("HSBG_LLM_URL")
        if not url:
            if backend == OLLAMA:
                url = DEFAULT_OLLAMA_URL
            else:
                raise LLMError(
                    f"HSBG_LLM_URL is required for backend={backend!r} — "
                    "there's no default endpoint for a hosted/OpenAI-"
                    "compatible server. Set it to e.g. "
                    "https://api.groq.com/openai or your LM Studio/llama.cpp "
                    "server's base URL."
                )
        api_key = os.environ.get("HSBG_LLM_KEY") or None
        timeout_raw = os.environ.get("HSBG_LLM_TIMEOUT")
        timeout_s = float(timeout_raw) if timeout_raw else DEFAULT_TIMEOUT_S
        return cls(backend=backend, model=model, url=url, api_key=api_key,
                   timeout_s=timeout_s)


class LLMClient:
    """One chat call, either backend, same call site.

    ``chat()`` is the only method callers need. Everything below it (URL
    building, payload shape, response parsing) is backend detail; the single
    network seam is ``_post`` so tests never touch a socket."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()

    def warmup(self, timeout_s: float = 120.0) -> bool:
        """Load the model into memory with one tiny request and a generous
        timeout. Ollama's COLD model load easily exceeds the per-turn
        timeout (observed live 2026-08-20: first call timed out at 8s on
        the owner's laptop) — after this, per-turn calls hit a warm model.
        Returns True when the model answered; never raises."""
        saved = self.config.timeout_s
        try:
            self.config.timeout_s = timeout_s
            self.chat("Reply with the single word: ok", "ok?", json_mode=False)
            return True
        except LLMError:
            return False
        finally:
            self.config.timeout_s = saved

    # --- the one network seam -----------------------------------------
    def _post(self, url: str, payload: dict, headers: Dict[str, str]) -> dict:
        """POST JSON, return parsed JSON. Raises LLMError on any HTTP/
        connection/timeout failure, with a message that names the likely fix."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise LLMError(self._http_error_message(exc.code, body)) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError) or "timed out" in str(exc.reason).lower():
                raise LLMError(self._timeout_message()) from exc
            raise LLMError(self._connect_error_message(str(exc.reason))) from exc
        except TimeoutError as exc:
            raise LLMError(self._timeout_message()) from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM server at {url} returned invalid JSON") from exc

    def _ollama_hint(self) -> str:
        return "is Ollama running? `ollama serve` / `ollama pull <model>`"

    def _http_error_message(self, code: int, body: str) -> str:
        hint = self._ollama_hint() if self.config.backend == OLLAMA else (
            "check HSBG_LLM_URL / HSBG_LLM_KEY / HSBG_LLM_MODEL for your "
            "OpenAI-compatible server")
        return f"LLM request failed ({code}): {body[:200]} — {hint}"

    def _connect_error_message(self, reason: str) -> str:
        if self.config.backend == OLLAMA:
            return (f"Could not reach Ollama at {self.config.url} ({reason}) — "
                    f"{self._ollama_hint()}")
        return f"Could not reach LLM server at {self.config.url} ({reason})"

    def _timeout_message(self) -> str:
        return (f"LLM request to {self.config.url} timed out after "
                f"{self.config.timeout_s}s — the model may be too slow for "
                "this backend; see llm-bench")

    # --- public API ------------------------------------------------------
    def chat(self, system: str, user: str, json_mode: bool = True) -> str:
        """One turn: system + user -> the model's raw text reply.

        `json_mode=True` (the Director's default) asks the server to
        constrain output to valid JSON; the caller still parses/validates it
        (`director._extract_json` + `validator.validate`) since "the server
        promised JSON" isn't the same as "the JSON is a legal move"."""
        if self.config.backend == OLLAMA:
            return self._chat_ollama(system, user, json_mode)
        return self._chat_openai(system, user, json_mode)

    def _chat_ollama(self, system: str, user: str, json_mode: bool) -> str:
        url = self.config.url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"num_predict": self.config.max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        resp = self._post(url, payload, {"Content-Type": "application/json"})
        content = (resp.get("message") or {}).get("content")
        if not content:
            raise LLMError(
                f"Ollama returned no message content: {json.dumps(resp)[:300]}")
        return content

    def _chat_openai(self, system: str, user: str, json_mode: bool) -> str:
        url = self.config.url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.config.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        resp = self._post(url, payload, headers)
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                "OpenAI-compatible response missing choices[0].message."
                f"content: {json.dumps(resp)[:300]}") from exc
        if not content:
            raise LLMError("OpenAI-compatible response had empty content")
        return content


def bench(client: LLMClient, prompt_chars: int = 4000) -> dict:
    """One realistic-size prompt, timed. Feeds the `llm-bench` CLI (wired up
    elsewhere) that decides local-vs-hosted per spec's "Open items" — the
    owner's ThinkPad numbers pick the deployment, not a guess.

    Returns {latency_s, output_chars, ok, error} — never raises; a failed
    call is a data point (ok=False), not a crash."""
    system = ("You are a fast Hearthstone Battlegrounds move summarizer. "
              "Reply in one short sentence.")
    user = ("Benchmark prompt (synthetic, latency measurement only) — "
            "current turn state:\n" + ("x" * max(0, prompt_chars - 80)))
    start = time.monotonic()
    try:
        out = client.chat(system, user, json_mode=False)
        latency = time.monotonic() - start
        return {"latency_s": round(latency, 3), "output_chars": len(out),
                "ok": True, "error": None}
    except LLMError as exc:
        latency = time.monotonic() - start
        return {"latency_s": round(latency, 3), "output_chars": 0,
                "ok": False, "error": str(exc)}
