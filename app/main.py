from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from typing import Any
from uuid import uuid4

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

from app.asr.parakeet_onnx import ParakeetOnnx
from app.aria_logging import get_logger
from app.audio.chunking import tail_window
from app.audio.ringbuffer import RingBuffer
from app.audio.vad import VadConfig, VoiceActivityDetector
from app.llm.ollama_client import generate_ollama_response, llm_enabled, load_ollama_config


log = get_logger("ARIA")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.asr = ParakeetOnnx()
        app.state.asr_error = None
        log.info("ARIA.ASR.Loaded", extra={"fields": {"model": "parakeet-tdt-0.6b-v3"}})
    except Exception as e:
        app.state.asr = None
        app.state.asr_error = repr(e)
        log.info("ARIA.ASR.NotLoaded", extra={"fields": {"error": repr(e)}})

    app.state.llm_client = None
    app.state.llm_config = None
    app.state.llm_queue = None
    app.state.llm_worker_task = None
    try:
        if llm_enabled():
            import httpx

            app.state.llm_config = load_ollama_config()
            app.state.llm_client = httpx.AsyncClient(timeout=app.state.llm_config.timeout_s)
            # Bounded queue prevents unbounded task accumulation; enqueue is non-blocking.
            app.state.llm_queue = asyncio.Queue(maxsize=1)
            app.state.llm_worker_task = asyncio.create_task(_llm_worker(app))
            log.info(
                "ARIA.LLM.Enabled",
                extra={
                    "fields": {
                        "provider": "ollama",
                        "url": app.state.llm_config.url,
                        "model": app.state.llm_config.model,
                    }
                },
            )
    except Exception as e:
        # Never fail server startup due to LLM wiring.
        app.state.llm_client = None
        app.state.llm_config = None
        log.info("ARIA.LLM.NotEnabled", extra={"fields": {"error": repr(e)}})

    yield

    worker = getattr(app.state, "llm_worker_task", None)
    if worker is not None:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    client = getattr(app.state, "llm_client", None)
    if client is not None:
        with contextlib.suppress(Exception):
            await client.aclose()


app = FastAPI(title="ARIA", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    asr_loaded = getattr(app.state, "asr", None) is not None
    return {
        "status": "ok",
        "asr_loaded": asr_loaded,
        "asr_error": getattr(app.state, "asr_error", None),
    }


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket) -> None:
    session_id = uuid4().hex
    await ws.accept()
    log.info("ARIA.Session.Connected", extra={"fields": {"session_id": session_id}})

    started = False
    buffering = False
    audio = RingBuffer(max_seconds=10.0, sample_rate=16000, channels=1)
    vad = VoiceActivityDetector(
        VadConfig(
            sample_rate=16000,
            frame_ms=20,
            aggressiveness=int(os.environ.get("ARIA_VAD_AGGR", "2")),
            silence_ms=int(os.environ.get("ARIA_VAD_SILENCE_MS", "700")),
        )
    )

    # Optional partial printing (server-side only).
    # If ARIA_PARTIAL_EVERY_MS=0, partials are disabled.
    partial_every_ms = int(os.environ.get("ARIA_PARTIAL_EVERY_MS", "0"))
    partial_every_s = max(0.0, partial_every_ms / 1000.0)
    partial_window_ms = int(os.environ.get("ARIA_PARTIAL_WINDOW_MS", "1500"))
    last_partial_at = 0.0
    partial_state: dict[str, str] = {"last_text": ""}

    utterance_gen = 0
    partial_task: asyncio.Task[None] | None = None

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                log.info("ARIA.Session.Disconnected", extra={"fields": {"session_id": session_id}})
                return

            text = message.get("text")
            data = message.get("bytes")

            if text is not None:
                payload = _safe_json(text)
                if payload is None:
                    continue

                msg_type = payload.get("type")
                if msg_type == "start":
                    if started:
                        await ws.close(code=1003)
                        return

                    if not _valid_start(payload):
                        log.info(
                            "ARIA.Session.InvalidStart",
                            extra={"fields": {"session_id": session_id, "payload": payload}},
                        )
                        await ws.close(code=1003)
                        return

                    started = True
                    log.info(
                        "ARIA.Session.Started",
                        extra={"fields": {"session_id": session_id, "chunk_ms": payload.get("chunk_ms")}},
                    )
                    continue

                if msg_type == "stop":
                    log.info("ARIA.Session.Stopped", extra={"fields": {"session_id": session_id}})
                    await ws.close()
                    return

                continue

            if data is not None:
                if not started:
                    await ws.close(code=1003)
                    return

                update = vad.push(data)

                if update.utterance_started:
                    utterance_gen += 1
                    audio.clear()
                    buffering = True
                    last_partial_at = time.monotonic()
                    partial_state["last_text"] = ""
                    log.info(
                        "ARIA.Utterance.Started",
                        extra={"fields": {"session_id": session_id, "utterance": utterance_gen}},
                    )

                if buffering:
                    dropped = audio.append(data)
                    if dropped:
                        log.info(
                            "ARIA.Audio.Dropped",
                            extra={
                                "fields": {
                                    "session_id": session_id,
                                    "utterance": utterance_gen,
                                    "dropped_bytes": dropped,
                                }
                            },
                        )

                    if partial_every_s > 0:
                        now = time.monotonic()
                        if now - last_partial_at >= partial_every_s:
                            last_partial_at = now
                            if partial_task is None or partial_task.done():
                                wav_tail = tail_window(
                                    audio.get_bytes(),
                                    sample_rate=16000,
                                    channels=1,
                                    window_ms=partial_window_ms,
                                )
                                partial_task = asyncio.create_task(
                                    _emit_partial_print(app, session_id, utterance_gen, wav_tail, partial_state)
                                )

                if update.utterance_ended and buffering:
                    wav_bytes = audio.get_bytes()
                    audio.clear()
                    buffering = False

                    if partial_task is not None:
                        partial_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await partial_task
                        partial_task = None

                    text_out = ""
                    asr: ParakeetOnnx | None = getattr(app.state, "asr", None)
                    if asr is not None and wav_bytes:
                        try:
                            text_out = await asyncio.to_thread(asr.transcribe, wav_bytes)
                        except Exception as e:
                            log.info(
                                "ARIA.ASR.Error",
                                extra={
                                    "fields": {
                                        "session_id": session_id,
                                        "utterance": utterance_gen,
                                        "error": repr(e),
                                    }
                                },
                            )
                            text_out = ""

                    # Spec change: server prints transcripts; nothing is sent back to clients.
                    if text_out:
                        print(f"ARIA.TRANSCRIPT: {text_out}", flush=True)
                        _enqueue_final_llm(app, text_out)
                    vad.reset()
                    log.info(
                        "ARIA.Utterance.Final",
                        extra={
                            "fields": {
                                "session_id": session_id,
                                "utterance": utterance_gen,
                                "text_len": len(text_out),
                                "bytes": len(wav_bytes),
                            }
                        },
                    )

    finally:
        if partial_task is not None:
            partial_task.cancel()
        pass


async def _emit_partial_print(
    app: FastAPI,
    session_id: str,
    utterance: int,
    wav_bytes: bytes,
    partial_state: dict[str, str],
) -> None:
    asr: ParakeetOnnx | None = getattr(app.state, "asr", None)
    if asr is None or not wav_bytes:
        return
    try:
        text_out = await asyncio.to_thread(asr.transcribe, wav_bytes)
    except Exception as e:
        log.info(
            "ARIA.ASR.PartialError",
            extra={"fields": {"session_id": session_id, "utterance": utterance, "error": repr(e)}},
        )
        return

    # Avoid printing the same partial repeatedly.
    if not text_out or text_out == partial_state.get("last_text", ""):
        return
    partial_state["last_text"] = text_out
    print(f"ARIA(partial) {text_out}", flush=True)


def _enqueue_final_llm(app: FastAPI, transcript: str) -> None:
    """Enqueue a final transcript for LLM processing without blocking."""

    q: asyncio.Queue[str] | None = getattr(app.state, "llm_queue", None)
    if q is None:
        return

    text = transcript.strip()
    if not text:
        return

    try:
        q.put_nowait(text)
    except asyncio.QueueFull:
        # Drop the previous queued item and keep the latest.
        with contextlib.suppress(Exception):
            _ = q.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            q.put_nowait(text)


async def _llm_worker(app: FastAPI) -> None:
    """Background worker that processes FINAL transcripts via local LLM.

    Runs independently of the WebSocket receive loop so LLM latency cannot block audio ingestion.
    """

    q: asyncio.Queue[str] | None = getattr(app.state, "llm_queue", None)
    config = getattr(app.state, "llm_config", None)
    client = getattr(app.state, "llm_client", None)
    if q is None or config is None:
        return

    while True:
        transcript = await q.get()
        try:
            response = await generate_ollama_response(text=transcript, config=config, client=client)
            if response:
                print(f"ARIA.LLM: {response}", flush=True)
        except Exception as e:
            log.info("ARIA.LLM.Error", extra={"fields": {"error": repr(e)}})


def _safe_json(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _valid_start(payload: dict[str, Any]) -> bool:
    return (
        payload.get("sample_rate") == 16000
        and payload.get("channels") == 1
        and payload.get("format") == "pcm_s16le"
        and isinstance(payload.get("chunk_ms"), int)
        and 20 <= payload.get("chunk_ms") <= 60
    )
