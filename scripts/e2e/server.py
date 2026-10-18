"""Test-only adapter: real Aria pipeline, captured Sonos output, no device plugins.

Run only via launch.sh in its private network namespace. Nothing here is imported
by production. The capture sink verifies actual HTTP delivery of normalized audio.
"""
from __future__ import annotations

import asyncio
import importlib
import io
import logging
import os
from pathlib import Path
import subprocess
import time
import wave

import httpx
import uvicorn
from fastapi.responses import FileResponse

RUN = Path(os.environ["ARIA_E2E_RUN"])
SOCKET = str(RUN / "aria.sock")
EVENTS: list[dict] = []


def record(name: str, /, **fields) -> None:
    EVENTS.append({"index": len(EVENTS), "time": time.time(), "event": name, **fields})


class EventHandler(logging.Handler):
    def emit(self, entry: logging.LogRecord) -> None:
        if entry.name.startswith("app") or str(entry.msg).startswith("ARIA."):
            record(str(entry.msg), **getattr(entry, "fields", {}))


class CaptureSink:
    def __init__(self, **_kwargs):
        pass

    async def set_volume(self, volume: int) -> None:
        record("E2E.VolumeCaptured", volume=volume)

    async def stop_all(self) -> None:
        record("E2E.StopCaptured")

    async def play_url(self, url: str) -> None:
        transport = httpx.AsyncHTTPTransport(uds=SOCKET)
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.get(url)
            response.raise_for_status()
        with wave.open(io.BytesIO(response.content)) as audio:
            frames = audio.getnframes()
            assert frames > 0, "Empty synthesized audio"
            record("E2E.AudioFetched", url=url, bytes=len(response.content),
                   rate=audio.getframerate(), channels=audio.getnchannels(),
                   duration=frames / audio.getframerate())


# Replace only the physical sink/discovery boundary, keeping SpeechOutput,
# PiperEngine, normalization, HTTP serving, echo guard, ASR, and LLM real.
speech = importlib.import_module("app.speech_output.speech_output")
speech.SonosHttpSink = CaptureSink
speech.verify_sonos_reachable = lambda *_args, **_kwargs: "E2E capture sink"
from app.main import app  # noqa: E402

logging.getLogger("ARIA").addHandler(EventHandler())


@app.get("/_e2e/state")
async def test_state(after: int = 0) -> dict:
    output = getattr(app.state, "speech_output", None)
    task = getattr(output, "_task", None)
    return {
        "asr": getattr(app.state, "asr", None) is not None,
        "llm": getattr(app.state, "llm_config", None) is not None,
        "tts": output is not None,
        "plugins": app.state.plugin_registry.plugin_count,
        "speaker_recognition": app.state.spk_recognizer.enabled,
        "router": app.state.router is not None,
        "speech_busy": bool(task and not task.done()),
        "events": EVENTS[max(0, after):],
        "next": len(EVENTS),
    }


@app.get("/_e2e/fixture/{language}")
async def fixture(language: str):
    phrases = {
        "en": ("Hello Aria, please tell me a short story about a little cat.", "en_US-amy-low"),
        "fr": ("Bonjour Aria, raconte-moi une courte histoire avec un petit chat.", "fr_FR-siwis-low"),
    }
    if language not in phrases:
        from fastapi import HTTPException
        raise HTTPException(404)
    text, voice = phrases[language]
    path = RUN / f"fixture-{language}.wav"
    if not path.exists():
        raw = await asyncio.to_thread(
            app.state.speech_output._engine.synthesize_to_wav, text,
            voice=f"/data/aria/voices/{voice}.onnx", rate=1.0,
        )
        await asyncio.to_thread(subprocess.run, [
            "/usr/bin/ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(raw),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(path),
        ], check=True, timeout=30)
    return FileResponse(path, media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, uds=SOCKET, log_level="info")
