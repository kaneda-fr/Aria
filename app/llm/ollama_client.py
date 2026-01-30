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
    "You are ARIA, a home automation assistant. When users request to control devices (lights, shutters, etc.), you MUST:\n"
    "1. Use find_devices to search for matching devices\n"
    "2. Use execute_command to control them\n"
    "Never claim to control devices without actually calling these tools. Always use tools for device control.",
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
    tools: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Send final transcript text to Ollama and return response.

    Supports both streaming and non-streaming modes via config.stream.
    When streaming, logs each chunk as received and returns accumulated text.

    Returns:
        Dict with 'content' (str) and optionally 'tool_calls' (list)
    """

    if messages is None:
        if not text.strip():
            return {"content": "", "tool_calls": None}
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text.strip()},
        ]

    async def _post(c: httpx.AsyncClient) -> dict[str, object]:
        # Prefer /api/chat for system+user prompting.
        chat_url = f"{config.url}/api/chat"
        payload = {
            "model": config.model,
            "stream": config.stream,
            "messages": messages,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools

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
            message = data.get("message") or {}
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls")

            return {
                "content": content.strip() if isinstance(content, str) else "",
                "tool_calls": tool_calls
            }

        # Streaming: parse newline-delimited JSON chunks
        async with c.stream("POST", chat_url, json=payload) as r:
            r.raise_for_status()
            accumulated = []
            tool_calls = None
            async for line in r.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    message = chunk.get("message") or {}
                    delta = message.get("content") or ""
                    if delta:
                        accumulated.append(delta)
                        print(f"ARIA.LLM.Chunk: {delta}", flush=True)
                    # Capture tool_calls from final chunk
                    if message.get("tool_calls"):
                        tool_calls = message.get("tool_calls")
                except json.JSONDecodeError:
                    continue
            return {
                "content": "".join(accumulated).strip(),
                "tool_calls": tool_calls
            }

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
    tools: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Send final transcript text to llama-cpp-python (OpenAI-compatible) and return response.

    Returns:
        Dict with 'content' (str) and optionally 'tool_calls' (list)
    """

    if messages is None:
        if not text.strip():
            return {"content": "", "tool_calls": None}
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

    # Add tools if provided (OpenAI format)
    if tools:
        payload["tools"] = tools

    if config.max_tokens is not None:
        payload["max_tokens"] = int(config.max_tokens)
    if config.temperature is not None:
        payload["temperature"] = float(config.temperature)
    if config.top_p is not None:
        payload["top_p"] = float(config.top_p)

    async def _post(c: httpx.AsyncClient) -> dict[str, object]:
        # Debug logging
        import os
        debug_enabled = os.environ.get("ARIA_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
        if debug_enabled:
            import json as json_lib
            print(f"[DEBUG] Sending to llama.cpp: {json_lib.dumps(payload, indent=2)}", flush=True)

        if not config.stream:
            r = await c.post(url, json=payload)
            r.raise_for_status()
            data = r.json()

            if debug_enabled:
                import json as json_lib
                print(f"[DEBUG] Received from llama.cpp: {json_lib.dumps(data, indent=2)}", flush=True)

            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                return {"content": "", "tool_calls": None}

            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls")

            return {
                "content": content.strip() if isinstance(content, str) else "",
                "tool_calls": tool_calls
            }

        async with c.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            accumulated: list[str] = []
            tool_calls = None
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
                delta = choices[0].get("delta") or {}
                content_delta = delta.get("content") or ""
                if content_delta:
                    accumulated.append(content_delta)
                    print(f"ARIA.LLM.Chunk: {content_delta}", flush=True)
                # Capture tool_calls from delta
                if delta.get("tool_calls"):
                    tool_calls = delta.get("tool_calls")
            return {
                "content": "".join(accumulated).strip(),
                "tool_calls": tool_calls
            }

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
    tools: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """
    Generate LLM response with optional tool calling support.

    Returns:
        Dict with 'content' (str) and optionally 'tool_calls' (list)
    """
    if isinstance(config, OllamaConfig):
        return await generate_ollama_response(text=text, messages=messages, config=config, client=client, tools=tools)
    return await generate_llamacpp_response(text=text, messages=messages, config=config, client=client, tools=tools)
