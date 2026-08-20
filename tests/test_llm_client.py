"""llm_client tests — offline only. The one network seam (`LLMClient._post`)
is monkeypatched for the payload-shape tests; the transport-error tests
monkeypatch `urllib.request.urlopen` itself (still no real socket) to prove
the HTTP/timeout -> LLMError mapping."""

import socket
import urllib.error

import pytest

from hsbg_coach import llm_client
from hsbg_coach.llm_client import LLMClient, LLMConfig, LLMError, bench


# --- LLMConfig.from_env -------------------------------------------------

def test_from_env_ollama_defaults(monkeypatch):
    monkeypatch.delenv("HSBG_LLM_BACKEND", raising=False)
    monkeypatch.delenv("HSBG_LLM_MODEL", raising=False)
    monkeypatch.delenv("HSBG_LLM_URL", raising=False)
    monkeypatch.delenv("HSBG_LLM_KEY", raising=False)
    monkeypatch.delenv("HSBG_LLM_TIMEOUT", raising=False)

    cfg = LLMConfig.from_env()
    assert cfg.backend == "ollama"
    assert cfg.model == "qwen2.5:3b-instruct"
    assert cfg.url == "http://localhost:11434"
    assert cfg.api_key is None
    assert cfg.timeout_s == 8


def test_from_env_reads_all_vars(monkeypatch):
    monkeypatch.setenv("HSBG_LLM_BACKEND", "openai")
    monkeypatch.setenv("HSBG_LLM_MODEL", "llama-3.1-8b")
    monkeypatch.setenv("HSBG_LLM_URL", "https://api.groq.com/openai")
    monkeypatch.setenv("HSBG_LLM_KEY", "sk-test")
    monkeypatch.setenv("HSBG_LLM_TIMEOUT", "12.5")

    cfg = LLMConfig.from_env()
    assert cfg.backend == "openai"
    assert cfg.model == "llama-3.1-8b"
    assert cfg.url == "https://api.groq.com/openai"
    assert cfg.api_key == "sk-test"
    assert cfg.timeout_s == 12.5


def test_from_env_openai_without_url_raises(monkeypatch):
    monkeypatch.setenv("HSBG_LLM_BACKEND", "openai")
    monkeypatch.delenv("HSBG_LLM_URL", raising=False)
    with pytest.raises(LLMError, match="HSBG_LLM_URL is required"):
        LLMConfig.from_env()


# --- payload shapes (monkeypatch the _post seam) ------------------------

def test_ollama_chat_payload_and_json_mode():
    client = LLMClient(LLMConfig(backend="ollama", model="qwen2.5:3b-instruct",
                                 url="http://localhost:11434", max_tokens=128))
    captured = {}

    def fake_post(url, payload, headers):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"message": {"content": '{"move": "Roll", "why": "ok"}'}}

    client._post = fake_post
    out = client.chat("sys prompt", "user prompt", json_mode=True)

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["model"] == "qwen2.5:3b-instruct"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["options"]["num_predict"] == 128
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert out == '{"move": "Roll", "why": "ok"}'


def test_ollama_chat_omits_format_when_not_json_mode():
    client = LLMClient(LLMConfig(backend="ollama"))
    captured = {}

    def fake_post(url, payload, headers):
        captured["payload"] = payload
        return {"message": {"content": "plain text reply"}}

    client._post = fake_post
    client.chat("sys", "user", json_mode=False)
    assert "format" not in captured["payload"]


def test_ollama_chat_raises_on_empty_content():
    client = LLMClient(LLMConfig(backend="ollama"))
    client._post = lambda url, payload, headers: {"message": {"content": ""}}
    with pytest.raises(LLMError, match="no message content"):
        client.chat("sys", "user")


def test_openai_chat_payload_and_auth_header():
    client = LLMClient(LLMConfig(backend="openai", model="llama-3.1-8b",
                                 url="https://api.groq.com/openai",
                                 api_key="sk-test", max_tokens=256))
    captured = {}

    def fake_post(url, payload, headers):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"choices": [{"message": {"content": '{"move": "End turn"}'}}]}

    client._post = fake_post
    out = client.chat("sys prompt", "user prompt", json_mode=True)

    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["payload"]["model"] == "llama-3.1-8b"
    assert captured["payload"]["max_tokens"] == 256
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert out == '{"move": "End turn"}'


def test_openai_chat_no_auth_header_without_key():
    client = LLMClient(LLMConfig(backend="openai", url="http://localhost:1234"))
    captured = {}

    def fake_post(url, payload, headers):
        captured["headers"] = headers
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    client._post = fake_post
    client.chat("sys", "user", json_mode=False)
    assert "Authorization" not in captured["headers"]
    assert "response_format" not in captured["payload"]


def test_openai_chat_raises_on_malformed_response():
    client = LLMClient(LLMConfig(backend="openai", url="http://localhost:1234"))
    client._post = lambda url, payload, headers: {"unexpected": "shape"}
    with pytest.raises(LLMError, match="missing choices"):
        client.chat("sys", "user")


# --- transport-level errors (monkeypatch urlopen, not _post) -----------

class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_post_maps_http_error_to_llm_error(monkeypatch):
    client = LLMClient(LLMConfig(backend="ollama", url="http://localhost:11434"))

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 500, "Internal Server Error",
            hdrs=None, fp=__import__("io").BytesIO(b"model not loaded"))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMError, match="failed \\(500\\)"):
        client.chat("sys", "user")


def test_post_maps_timeout_to_llm_error(monkeypatch):
    client = LLMClient(LLMConfig(backend="ollama", url="http://localhost:11434",
                                 timeout_s=1))

    def fake_urlopen(req, timeout=None):
        raise socket.timeout("timed out")

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMError, match="timed out"):
        client.chat("sys", "user")


def test_post_maps_connection_refused_to_llm_error(monkeypatch):
    client = LLMClient(LLMConfig(backend="ollama", url="http://localhost:11434"))

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError("Connection refused"))

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMError, match="Could not reach Ollama"):
        client.chat("sys", "user")


def test_post_succeeds_through_real_urlopen_seam(monkeypatch):
    """Confirms _post actually parses a well-formed response via the real
    urlopen call path (still offline: urlopen itself is faked)."""
    client = LLMClient(LLMConfig(backend="ollama", url="http://localhost:11434"))

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(b'{"message": {"content": "hi"}}')

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)
    out = client.chat("sys", "user", json_mode=False)
    assert out == "hi"


# --- bench() -------------------------------------------------------------

def test_bench_ok_path():
    client = LLMClient(LLMConfig(backend="ollama"))
    client.chat = lambda system, user, json_mode=True: "a short reply"
    result = bench(client, prompt_chars=500)
    assert result["ok"] is True
    assert result["error"] is None
    assert result["output_chars"] == len("a short reply")
    assert result["latency_s"] >= 0


def test_bench_error_path_never_raises():
    client = LLMClient(LLMConfig(backend="ollama"))

    def raising_chat(system, user, json_mode=True):
        raise LLMError("is Ollama running? `ollama serve`")

    client.chat = raising_chat
    result = bench(client)
    assert result["ok"] is False
    assert result["output_chars"] == 0
    assert "Ollama running" in result["error"]
