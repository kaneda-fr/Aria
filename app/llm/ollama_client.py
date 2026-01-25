from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal


SYSTEM_PROMPT = os.environ.get(
    "ARIA_LLM_SYSTEM_PROMPT",
    "You are ARIA, a local assistant running on a private server. Answer in 1-2 sentences. Be concise and direct.",
)


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    provider: Literal["ollama"]
    url: str
    model: str
    timeout_s: float = 12.0
    num_predict: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


@dataclass(frozen=True, slots=True)
class LlamaCppConfig:
    provider: Literal["llamacpp"]
    url: str  # OpenAI-compatible base URL, typically http://127.0.0.1:8001/v1
    model: str = "local"
    timeout_s: float = 12.0
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


def llm_enabled() -> bool:
    provider = os.environ.get("ARIA_LLM_PROVIDER", "").strip().lower()
    if provider in ("", "0", "false", "off", "none", "disabled"):
        return False
    return provider in {"ollama", "llamacpp"}


def llm_provider() -> str | None:
    provider = os.environ.get("ARIA_LLM_PROVIDER", "").strip().lower()
    if provider in ("", "0", "false", "off", "none", "disabled"):
        return None
    return provider


def _opt_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _opt_float(name: str) -> float | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_ollama_config() -> OllamaConfig:
    url = os.environ.get("ARIA_LLM_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("ARIA_LLM_MODEL", "qwen2.5:3b-instruct")
    timeout_s = float(os.environ.get("ARIA_LLM_TIMEOUT_S", "12"))

    num_predict = _opt_int("ARIA_LLM_NUM_PREDICT")
    temperature = _opt_float("ARIA_LLM_TEMPERATURE")
    top_p = _opt_float("ARIA_LLM_TOP_P")
    stream = os.environ.get("ARIA_LLM_STREAM", "0").strip().lower() in ("1", "true", "yes", "on")

    return OllamaConfig(
        provider="ollama",
        url=url,
        model=model,
        timeout_s=timeout_s,
        num_predict=num_predict,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
    )


def load_llamacpp_config() -> LlamaCppConfig:
    # llama-cpp-python server exposes an OpenAI-compatible API under /v1.
    # Example: http://127.0.0.1:8001/v1
    url = os.environ.get("ARIA_LLM_URL", "http://127.0.0.1:8001/v1").rstrip("/")
    model = os.environ.get("ARIA_LLM_MODEL", "local")
    timeout_s = float(os.environ.get("ARIA_LLM_TIMEOUT_S", "12"))

    max_tokens = _opt_int("ARIA_LLM_NUM_PREDICT")
    temperature = _opt_float("ARIA_LLM_TEMPERATURE")
    top_p = _opt_float("ARIA_LLM_TOP_P")
    stream = os.environ.get("ARIA_LLM_STREAM", "0").strip().lower() in ("1", "true", "yes", "on")

    return LlamaCppConfig(
        provider="llamacpp",
        url=url,
        model=model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
    )


def load_llm_config() -> OllamaConfig | LlamaCppConfig:
    provider = llm_provider()
    if provider == "ollama":
        return load_ollama_config()
    if provider == "llamacpp":
        return load_llamacpp_config()
    raise ValueError(f"Unsupported ARIA_LLM_PROVIDER: {provider!r}")


async def generate_ollama_response(
    *,
    text: str,
    messages: list[dict[str, str]] | None = None,
    config: OllamaConfig,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Send final transcript text to Ollama and return response text.

    Supports both streaming and non-streaming modes via config.stream.
    When streaming, logs each chunk as received and returns accumulated text.
    """

    if messages is None:
        if not text.strip():
            return ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text.strip()},
        ]

    async def _post(c: httpx.AsyncClient) -> str:
        # Prefer /api/chat for system+user prompting.
        chat_url = f"{config.url}/api/chat"
        payload = {
            "model": config.model,
            "stream": config.stream,
            "messages": messages,
        }

        options: dict[str, object] = {}
        if config.num_predict is not None:
            options["num_predict"] = int(config.num_predict)
        if config.temperature is not None:
            options["temperature"] = float(config.temperature)
        if config.top_p is not None:
            options["top_p"] = float(config.top_p)
        if options:
            payload["options"] = options

        if not config.stream:
            # Non-streaming: simple JSON response
            r = await c.post(chat_url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = (data.get("message") or {}).get("content")
            if isinstance(msg, str):
                return msg.strip()
            return ""

        # Streaming: parse newline-delimited JSON chunks
        async with c.stream("POST", chat_url, json=payload) as r:
            r.raise_for_status()
            accumulated = []
            async for line in r.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    delta = (chunk.get("message") or {}).get("content") or ""
                    if delta:
                        accumulated.append(delta)
                        print(f"ARIA.LLM.Chunk: {delta}", flush=True)
                except json.JSONDecodeError:
                    continue
            return "".join(accumulated).strip()

    if client is not None:
        return await _post(client)

    timeout = httpx.Timeout(config.timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as c:
        return await _post(c)


async def generate_llamacpp_response(
    *,
    text: str,
    messages: list[dict[str, str]] | None = None,
    config: LlamaCppConfig,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Send final transcript text to llama-cpp-python (OpenAI-compatible) and return response text."""

    if messages is None:
        if not text.strip():
            return ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text.strip()},
        ]

    # Allow url to be either .../v1 or .../v1/ (or even .../v1/chat/completions).
    base = config.url.rstrip("/")
    if base.endswith("/chat/completions"):
        url = base
    else:
        url = f"{base}/chat/completions"

    payload: dict[str, object] = {
        "model": config.model,
        "stream": config.stream,
        "messages": messages,
    }
    if config.max_tokens is not None:
        payload["max_tokens"] = int(config.max_tokens)
    if config.temperature is not None:
        payload["temperature"] = float(config.temperature)
    if config.top_p is not None:
        payload["top_p"] = float(config.top_p)

    async def _post(c: httpx.AsyncClient) -> str:
        if not config.stream:
            r = await c.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                return ""
            msg = (choices[0].get("message") or {}).get("content")
            if isinstance(msg, str):
                return msg.strip()
            return ""

        async with c.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            accumulated: list[str] = []
            async for line in r.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta:
                    accumulated.append(delta)
                    print(f"ARIA.LLM.Chunk: {delta}", flush=True)
            return "".join(accumulated).strip()

    if client is not None:
        return await _post(client)

    timeout = httpx.Timeout(config.timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as c:
        return await _post(c)


async def generate_llm_response(
    *,
    text: str,
    messages: list[dict[str, str]] | None = None,
    config: OllamaConfig | LlamaCppConfig,
    client: httpx.AsyncClient | None = None,
) -> str:
    if isinstance(config, OllamaConfig):
        return await generate_ollama_response(text=text, messages=messages, config=config, client=client)
    return await generate_llamacpp_response(text=text, messages=messages, config=config, client=client)
