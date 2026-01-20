from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


SYSTEM_PROMPT = "You are ARIA, a local assistant running on a private server."


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    url: str
    model: str
    timeout_s: float = 12.0


def llm_enabled() -> bool:
    provider = os.environ.get("ARIA_LLM_PROVIDER", "").strip().lower()
    if provider in ("", "0", "false", "off", "none", "disabled"):
        return False
    return provider == "ollama"


def load_ollama_config() -> OllamaConfig:
    url = os.environ.get("ARIA_LLM_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("ARIA_LLM_MODEL", "qwen2.5:3b-instruct")
    timeout_s = float(os.environ.get("ARIA_LLM_TIMEOUT_S", "12"))
    return OllamaConfig(url=url, model=model, timeout_s=timeout_s)


async def generate_ollama_response(
    *,
    text: str,
    config: OllamaConfig,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Send final transcript text to Ollama and return response text.

    Uses Ollama's HTTP API and does not stream.
    """

    if not text.strip():
        return ""

    async def _post(c: httpx.AsyncClient) -> str:
        # Prefer /api/chat for system+user prompting.
        chat_url = f"{config.url}/api/chat"
        payload = {
            "model": config.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text.strip()},
            ],
        }
        r = await c.post(chat_url, json=payload)
        r.raise_for_status()
        data = r.json()
        msg = (data.get("message") or {}).get("content")
        if isinstance(msg, str):
            return msg.strip()
        return ""

    if client is not None:
        return await _post(client)

    timeout = httpx.Timeout(config.timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as c:
        return await _post(c)
