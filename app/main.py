from __future__ import annotations

import asyncio
import contextlib
import audioop
import json
import os
import time
import shutil
from collections import deque
from typing import Any
from uuid import uuid4

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

from app.asr.parakeet_onnx import ParakeetOnnx
from app.aria_logging import get_logger
from app.audio.chunking import tail_window
from app.audio.ringbuffer import RingBuffer
from app.audio.vad import VadConfig, VoiceActivityDetector
from app.http_audio.server import build_tts_router
from app.llm.ollama_client import SYSTEM_PROMPT, generate_llm_response, llm_enabled, load_llm_config
from app.plugins.registry import PluginRegistry
from app.plugins.inventory import InventoryProvider
from app.plugins.actions import ActionsProvider
from app.speaker_recognition.recognizer import SpeakerRecognizer
from app.speech_output.echo_guard_v2 import EchoGuardV2
from app.speech_output.speech_output import maybe_build_speech_output


def _pcm16_stats(pcm16le: bytes, *, sample_rate: int = 16000) -> tuple[float, float]:
    if not pcm16le:
        return 0.0, 0.0
    seconds = len(pcm16le) / (2.0 * float(sample_rate))
    rms = float(audioop.rms(pcm16le, 2)) / 32768.0
    return seconds, rms


def _webrtc_speech_ratio(
    pcm16le: bytes,
    *,
    sample_rate: int = 16000,
    frame_ms: int = 20,
    aggressiveness: int = 2,
) -> float:
    try:
        import webrtcvad

        vad = webrtcvad.Vad(int(aggressiveness))
    except Exception:
        return 1.0

    frame_bytes = int(sample_rate * 2 * frame_ms / 1000)
    if frame_bytes <= 0 or len(pcm16le) < frame_bytes:
        return 0.0

    total = 0
    speech = 0
    end = len(pcm16le) - (len(pcm16le) % frame_bytes)
    for off in range(0, end, frame_bytes):
        frame = pcm16le[off : off + frame_bytes]
        total += 1
        try:
            if vad.is_speech(frame, sample_rate):
                speech += 1
        except Exception:
            continue
    return (speech / total) if total else 0.0


def _should_run_asr(pcm16le: bytes) -> tuple[bool, dict[str, float]]:
    min_seconds = float(os.environ.get("ARIA_ASR_MIN_SECONDS", "0.30"))
    min_rms = float(os.environ.get("ARIA_ASR_MIN_RMS", "0.004"))
    min_speech_ratio = float(os.environ.get("ARIA_ASR_MIN_SPEECH_RATIO", "0.18"))
    aggr = int(os.environ.get("ARIA_VAD_AGGR", "2"))

    seconds, rms = _pcm16_stats(pcm16le)
    speech_ratio = _webrtc_speech_ratio(pcm16le, aggressiveness=aggr)
    metrics = {"seconds": seconds, "rms": rms, "speech_ratio": speech_ratio}

    if seconds < max(0.0, min_seconds):
        return False, metrics
    if rms < max(0.0, min_rms):
        return False, metrics
    if min_speech_ratio > 0.0 and speech_ratio < min_speech_ratio:
        return False, metrics
    return True, metrics


def _filter_transcript(text: str, *, seconds: float, speech_ratio: float) -> str:
    if not text:
        return ""

    if os.environ.get("ARIA_ASR_DROP_FILLERS", "1").strip().lower() in {"0", "false", "no", "off"}:
        return text

    fillers = {
        "yeah",
        "yea",
        "yep",
        "ok",
        "okay",
        "oh",
        "uh",
        "um",
        "hmm",
        "mm",
        "mhm",
    }

    norm = " ".join(text.strip().lower().split())
    if norm.rstrip(".!?") in fillers:
        # Only drop if it looks like low-speech / noise-triggered.
        if seconds < float(os.environ.get("ARIA_ASR_FILLER_MAX_SECONDS", "0.9")) and speech_ratio < float(
            os.environ.get("ARIA_ASR_FILLER_MIN_SPEECH_RATIO", "0.35")
        ):
            return ""
    return text


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
    app.state.llm_history = {}
    app.state.llm_history_lock = asyncio.Lock()
    app.state.speech_output = None
    app.state.tts_cache_dir = None
    app.state.echo_guard = EchoGuardV2()
    app.state.spk_recognizer = SpeakerRecognizer()
    try:
        if llm_enabled():
            import httpx

            app.state.llm_config = load_llm_config()
            app.state.llm_client = httpx.AsyncClient(timeout=app.state.llm_config.timeout_s)
            # Bounded queue prevents unbounded task accumulation; enqueue is non-blocking.
            app.state.llm_queue = asyncio.Queue(maxsize=1)
            app.state.llm_worker_task = asyncio.create_task(_llm_worker(app))
            cfg = app.state.llm_config
            log.info(
                "ARIA.LLM.Enabled",
                extra={
                    "fields": {
                        "provider": getattr(cfg, "provider", None),
                        "url": getattr(cfg, "url", None) or getattr(cfg, "base_url", None),
                        "model": getattr(cfg, "model", None),
                    }
                },
            )
    except Exception as e:
        # Never fail server startup due to LLM wiring.
        app.state.llm_client = None
        app.state.llm_config = None
        log.info("ARIA.LLM.NotEnabled", extra={"fields": {"error": repr(e)}})

    # TTS / Sonos speech output (optional)
    try:
        # Validate required local binaries before enabling TTS.
        piper_bin = os.environ.get("ARIA_PIPER_BIN", "piper").strip() or "piper"
        ffmpeg_bin = os.environ.get("ARIA_FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"
        missing: list[str] = []
        if shutil.which(piper_bin) is None:
            missing.append(piper_bin)
        if shutil.which(ffmpeg_bin) is None:
            missing.append(ffmpeg_bin)

        if missing:
            log.info(
                "ARIA.TTS.NotEnabled",
                extra={"fields": {"error": f"missing binaries: {', '.join(missing)}"}},
            )
            speech = None
        else:
            speech = maybe_build_speech_output(app.state.echo_guard)
        if speech is not None:
            app.state.speech_output = speech
            app.state.tts_cache_dir = speech.tts_cache_dir()
            log.info(
                "ARIA.TTS.Enabled",
                extra={
                    "fields": {
                        "engine": getattr(speech.cfg, "engine", None),
                        "cache_dir": str(speech.tts_cache_dir()),
                    }
                },
            )
    except Exception as e:
        app.state.speech_output = None
        app.state.tts_cache_dir = None
        log.info("ARIA.TTS.NotEnabled", extra={"fields": {"error": repr(e)}})

    # Plugin System (optional, enabled via env)
    app.state.plugin_registry = PluginRegistry()
    try:
        plugins_enabled = os.environ.get("ARIA_PLUGINS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
        if plugins_enabled:
            # Register plugin classes
            app.state.plugin_registry.register_plugin_class("inventory", InventoryProvider)
            app.state.plugin_registry.register_plugin_class("actions", ActionsProvider)

            # Initialize Inventory Provider if enabled
            inventory_enabled = os.environ.get("ARIA_INVENTORY_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
            if inventory_enabled:
                # Build inventory config from env
                inventory_config = {
                    "cache_path": os.environ.get("ARIA_INVENTORY_CACHE_PATH", "/tmp/aria_inventory.json"),
                    "sources": {
                        "jeedom": {
                            "type": "jeedom_mqtt",
                            "host": os.environ.get("ARIA_JEEDOM_MQTT_HOST", "localhost"),
                            "port": int(os.environ.get("ARIA_JEEDOM_MQTT_PORT", "1883")),
                            "username": os.environ.get("ARIA_JEEDOM_MQTT_USERNAME"),
                            "password": os.environ.get("ARIA_JEEDOM_MQTT_PASSWORD"),
                            "root_topic": os.environ.get("ARIA_JEEDOM_MQTT_ROOT_TOPIC", "jeedom"),
                            "qos": int(os.environ.get("ARIA_MQTT_QOS_SUBSCRIBE", "1")),
                        }
                    }
                }

                # Create and start inventory provider
                app.state.plugin_registry.create_plugin("inventory", inventory_config)

            # Initialize Actions Provider if enabled
            actions_enabled = os.environ.get("ARIA_ACTIONS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
            if actions_enabled:
                # Build actions config from env (shares MQTT config with inventory)
                actions_config = {
                    "host": os.environ.get("ARIA_JEEDOM_MQTT_HOST", "localhost"),
                    "port": int(os.environ.get("ARIA_JEEDOM_MQTT_PORT", "1883")),
                    "username": os.environ.get("ARIA_JEEDOM_MQTT_USERNAME"),
                    "password": os.environ.get("ARIA_JEEDOM_MQTT_PASSWORD"),
                    "root_topic": os.environ.get("ARIA_JEEDOM_MQTT_ROOT_TOPIC", "jeedom"),
                    "qos": int(os.environ.get("ARIA_MQTT_QOS_PUBLISH", "1")),
                }

                # Create and start actions provider
                app.state.plugin_registry.create_plugin("actions", actions_config)

            log.info("ARIA.Plugins.Enabled", extra={"fields": {"plugin_count": app.state.plugin_registry.plugin_count}})
            # Start all registered plugins
            app.state.plugin_registry.start_all()
        else:
            log.info("ARIA.Plugins.Disabled", extra={"fields": {}})
    except Exception as e:
        log.info("ARIA.Plugins.NotEnabled", extra={"fields": {"error": repr(e)}})

    yield

    # Cleanup: Stop plugins first (before other resources)
    plugin_registry = getattr(app.state, "plugin_registry", None)
    if plugin_registry is not None:
        try:
            log.info("ARIA.Plugins.Stopping")
            plugin_registry.stop_all()
            log.info("ARIA.Plugins.Stopped")
        except Exception as e:
            log.error("ARIA.Plugins.StopError", extra={"fields": {"error": repr(e)}})

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

# Expose cached/normalized audio clips for Sonos to fetch.
# This is safe to mount even when TTS is disabled; requests will 404.
try:
    import pathlib

    cache_dir = pathlib.Path(os.environ.get("ARIA_TTS_CACHE_DIR", "/tmp/aria_tts_cache"))
    app.include_router(build_tts_router(tts_cache_dir=cache_dir))
except Exception:
    pass


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    asr_loaded = getattr(app.state, "asr", None) is not None

    # Get plugin health if registry exists
    plugin_health = None
    plugin_registry = getattr(app.state, "plugin_registry", None)
    if plugin_registry is not None and plugin_registry.plugin_count > 0:
        plugin_health = plugin_registry.health_all()

    response = {
        "status": "ok",
        "asr_loaded": asr_loaded,
        "asr_error": getattr(app.state, "asr_error", None),
    }

    if plugin_health is not None:
        response["plugins"] = plugin_health

    return response


@app.get("/api/inventory/snapshot")
async def get_inventory_snapshot() -> dict[str, Any]:
    """Get complete inventory snapshot."""
    from fastapi import HTTPException
    from app.plugins.tooling_provider import ToolingProvider

    plugin_registry = getattr(app.state, "plugin_registry", None)
    if plugin_registry is None:
        raise HTTPException(status_code=503, detail="Plugin system not enabled")

    # Get inventory provider
    providers = plugin_registry.get_plugins_by_type(ToolingProvider)
    inventory = next((p for p in providers if p.name == "inventory"), None)

    if inventory is None:
        raise HTTPException(status_code=503, detail="Inventory provider not available")

    if not inventory.is_started:
        raise HTTPException(status_code=503, detail="Inventory provider not started")

    return inventory.get_snapshot()


@app.post("/api/inventory/candidates")
async def get_inventory_candidates(request: dict[str, Any]) -> dict[str, Any]:
    """
    Get command candidates matching user input.

    Request body:
        {
            "text": str,              # Required: User input text
            "context": dict | None    # Optional: Context (room, speaker, etc.)
        }
    """
    from fastapi import HTTPException
    from app.plugins.tooling_provider import ToolingProvider

    plugin_registry = getattr(app.state, "plugin_registry", None)
    if plugin_registry is None:
        raise HTTPException(status_code=503, detail="Plugin system not enabled")

    # Get inventory provider
    providers = plugin_registry.get_plugins_by_type(ToolingProvider)
    inventory = next((p for p in providers if p.name == "inventory"), None)

    if inventory is None:
        raise HTTPException(status_code=503, detail="Inventory provider not available")

    if not inventory.is_started:
        raise HTTPException(status_code=503, detail="Inventory provider not started")

    # Extract parameters
    text = request.get("text")
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Missing or invalid 'text' parameter")

    context = request.get("context")
    if context is not None and not isinstance(context, dict):
        raise HTTPException(status_code=400, detail="Invalid 'context' parameter (must be dict)")

    return inventory.get_candidates(text=text, context=context)


@app.post("/api/actions/execute")
async def execute_action(request: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a command on a device.

    Request body:
        {
            "cmd_id": str,         # Required: Command ID to execute
            "value": any | None    # Optional: Value for the command (e.g., 50 for dimmer)
        }
    """
    from fastapi import HTTPException
    from app.plugins.tooling_provider import ToolingProvider
    from app.plugins.actions import ActionsProvider

    plugin_registry = getattr(app.state, "plugin_registry", None)
    if plugin_registry is None:
        raise HTTPException(status_code=503, detail="Plugin system not enabled")

    # Get actions provider
    actions = plugin_registry.get_plugin("actions")
    if actions is None or not isinstance(actions, ActionsProvider):
        raise HTTPException(status_code=503, detail="Actions provider not available")

    if not actions.is_started:
        raise HTTPException(status_code=503, detail="Actions provider not started")

    # Extract parameters
    cmd_id = request.get("cmd_id")
    if not cmd_id or not isinstance(cmd_id, str):
        raise HTTPException(status_code=400, detail="Missing or invalid 'cmd_id' parameter")

    value = request.get("value")

    # Get execution spec from inventory (optional)
    execution_spec = None
    providers = plugin_registry.get_plugins_by_type(ToolingProvider)
    inventory = next((p for p in providers if p.name == "inventory"), None)
    if inventory and inventory.is_started:
        # Lookup command in inventory to get execution spec
        cmd = inventory.indexer.get_command(cmd_id)
        if cmd and cmd.execution:
            execution_spec = cmd.execution

    # Execute command
    result = actions.execute_command(cmd_id=cmd_id, value=value, execution_spec=execution_spec)
    return result


@app.post("/api/llm/process")
async def process_text(request: dict[str, Any]) -> dict[str, Any]:
    """
    Process text through the LLM pipeline (bypassing ASR) with tool calling.

    Request body:
        {
            "text": str,                    # Required: Text to process
            "session_id": str | None,       # Optional: Session ID for history
            "speaker_user": str | None,     # Optional: Speaker identifier (default: "unknown")
            "speak_response": bool,         # Optional: Speak response via TTS (default: false)
            "use_tools": bool               # Optional: Enable tool calling (default: true)
        }

    Response:
        {
            "success": bool,
            "text": str,                    # Input text
            "response": str,                # LLM response
            "error": str | None,
            "spoken": bool,                 # Whether response was spoken via TTS
            "tool_calls_made": int          # Number of tool calls executed
        }
    """
    from fastapi import HTTPException
    from app.llm.tools import TOOLS, execute_tool, format_tool_result_for_llm
    from app.plugins.tooling_provider import ToolingProvider
    from app.plugins.actions import ActionsProvider

    # Extract parameters
    text = request.get("text")
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Missing or invalid 'text' parameter")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    session_id = request.get("session_id")
    speaker_user = request.get("speaker_user", "unknown")
    speak_response = request.get("speak_response", False)
    use_tools = request.get("use_tools", True)

    # Check if LLM is enabled
    llm_config = getattr(app.state, "llm_config", None)
    llm_client = getattr(app.state, "llm_client", None)

    if not llm_config or not llm_client:
        raise HTTPException(status_code=503, detail="LLM not enabled")

    # Get plugin providers for tools
    plugin_registry = getattr(app.state, "plugin_registry", None)
    inventory_provider = None
    actions_provider = None

    if use_tools and plugin_registry:
        providers = plugin_registry.get_plugins_by_type(ToolingProvider)
        inventory_provider = next((p for p in providers if p.name == "inventory"), None)
        actions_provider = plugin_registry.get_plugin("actions")

    try:
        # Build messages with optional history
        history_max = _llm_history_max_messages()
        messages: list[dict[str, Any]] = []

        if history_max > 0 and session_id:
            history_lock = getattr(app.state, "llm_history_lock", None)
            history_map = getattr(app.state, "llm_history", None)
            if history_lock is not None and isinstance(history_map, dict):
                async with history_lock:
                    history = history_map.get(session_id)
                    if history:
                        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *list(history)]

        if not messages:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        user_content = f"[speaker={speaker_user}] {text}"
        messages.append({"role": "user", "content": user_content})

        # Tool calling loop
        tool_calls_made = 0
        max_iterations = 5
        t0 = time.monotonic()
        final_response = ""

        for iteration in range(max_iterations):
            # Generate LLM response with tools
            llm_result = await generate_llm_response(
                text=text,
                messages=messages,
                config=llm_config,
                client=llm_client,
                tools=TOOLS if use_tools else None,
            )

            response_content = llm_result.get("content", "")
            tool_calls = llm_result.get("tool_calls")

            # If no tool calls, we have the final response
            if not tool_calls:
                final_response = response_content
                break

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": response_content or "",
                "tool_calls": tool_calls
            })

            # Execute each tool call
            for tool_call in tool_calls:
                tool_calls_made += 1

                # Extract tool call details
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")
                tool_args_raw = function.get("arguments", {})

                # Handle both Ollama (dict) and OpenAI (string) formats
                if isinstance(tool_args_raw, dict):
                    tool_args = tool_args_raw
                elif isinstance(tool_args_raw, str):
                    try:
                        tool_args = json.loads(tool_args_raw)
                    except json.JSONDecodeError:
                        tool_args = {}
                else:
                    tool_args = {}

                # Execute tool
                tool_result = await execute_tool(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    inventory_provider=inventory_provider,
                    actions_provider=actions_provider
                )

                # Format result for LLM
                tool_result_text = format_tool_result_for_llm(tool_result)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": tool_result_text,
                    "name": tool_name
                })

        dt_ms = int((time.monotonic() - t0) * 1000)

        # Check if we got a response
        if not final_response and iteration >= max_iterations - 1:
            log.warning(
                "ARIA.LLM.MaxIterationsReached",
                extra={"fields": {"ms": dt_ms, "iterations": max_iterations, "tool_calls_made": tool_calls_made}}
            )
            return {
                "success": False,
                "text": text,
                "response": "I tried to help but ran into too many steps. Please try again.",
                "error": "Max tool calling iterations reached",
                "spoken": False,
                "tool_calls_made": tool_calls_made
            }

        if not final_response:
            log.info(
                "ARIA.LLM.EmptyResponse",
                extra={"fields": {"ms": dt_ms, "transcript_len": len(text), "source": "api", "tool_calls_made": tool_calls_made}}
            )
            return {
                "success": False,
                "text": text,
                "response": "",
                "error": "Empty response from LLM",
                "spoken": False,
                "tool_calls_made": tool_calls_made
            }

        log.info(
            "ARIA.LLM.Done",
            extra={
                "fields": {
                    "ms": dt_ms,
                    "transcript_len": len(text),
                    "response_len": len(final_response),
                    "tool_calls_made": tool_calls_made,
                    "source": "api"
                }
            },
        )

        # Update history if session_id provided
        if history_max > 0 and session_id:
            history_lock = getattr(app.state, "llm_history_lock", None)
            history_map = getattr(app.state, "llm_history", None)
            if history_lock is not None and isinstance(history_map, dict):
                async with history_lock:
                    history = history_map.get(session_id)
                    if history is None or getattr(history, "maxlen", None) != history_max:
                        history = deque(list(history or []), maxlen=history_max)
                        history_map[session_id] = history
                    history.append({"role": "user", "content": user_content})
                    history.append({"role": "assistant", "content": final_response})

        # Optional TTS
        spoken = False
        if speak_response:
            speech = getattr(app.state, "speech_output", None)
            if speech is not None:
                try:
                    await speech.speak(final_response)
                    spoken = True
                except Exception as e:
                    log.info("ARIA.TTS.Error", extra={"fields": {"error": repr(e)}})

        return {
            "success": True,
            "text": text,
            "response": final_response,
            "error": None,
            "spoken": spoken,
            "tool_calls_made": tool_calls_made
        }

    except Exception as e:
        log.error(
            "ARIA.LLM.Error",
            extra={"fields": {"error": repr(e), "transcript_len": len(text), "source": "api"}},
        )
        raise HTTPException(status_code=500, detail=f"LLM processing error: {str(e)}")


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket) -> None:
    session_id = uuid4().hex
    await ws.accept()
    log.info("ARIA.Session.Connected", extra={"fields": {"session_id": session_id}})

    started = False
    client_vad_enabled = False
    buffering = False
    client_vad_started = False
    dropped_prestart_bytes = 0
    audio = RingBuffer(max_seconds=10.0, sample_rate=16000, channels=1)
    vad = VoiceActivityDetector(
        VadConfig(
            sample_rate=16000,
            frame_ms=20,
            aggressiveness=int(os.environ.get("ARIA_VAD_AGGR", "2")),
            start_frames=int(os.environ.get("ARIA_VAD_START_FRAMES", "3")),
            min_rms=float(os.environ.get("ARIA_VAD_MIN_RMS", "0.0")),
            snr_db=float(os.environ.get("ARIA_VAD_SNR_DB", "0.0")),
            noise_alpha=float(os.environ.get("ARIA_VAD_NOISE_ALPHA", "0.05")),
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

    async def _start_utterance() -> None:
        nonlocal buffering, utterance_gen, last_partial_at
        utterance_gen += 1
        audio.clear()
        buffering = True
        last_partial_at = time.monotonic()
        partial_state["last_text"] = ""
        log.info(
            "ARIA.Utterance.Started",
            extra={"fields": {"session_id": session_id, "utterance": utterance_gen}},
        )

    async def _finalize_utterance(*, reason: str) -> None:
        nonlocal buffering, partial_task
        if not buffering:
            return

        wav_bytes = audio.get_bytes()
        audio.clear()
        buffering = False

        should_asr, asr_metrics = _should_run_asr(wav_bytes)

        if partial_task is not None:
            partial_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await partial_task
            partial_task = None

        text_out = ""
        asr: ParakeetOnnx | None = getattr(app.state, "asr", None)
        if asr is not None and wav_bytes and should_asr:
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

        # Drop common filler hallucinations for low-speech utterances.
        text_out = _filter_transcript(
            text_out,
            seconds=float(asr_metrics.get("seconds", 0.0)),
            speech_ratio=float(asr_metrics.get("speech_ratio", 0.0)),
        )

        # Spec change: server prints transcripts; nothing is sent back to clients.

        should_enqueue = True
        speaker_user = "unknown"
        if text_out:
            spk = getattr(app.state, "spk_recognizer", None)
            if spk is not None and getattr(spk, "enabled", False):
                timeout_s = float(os.environ.get("ARIA_SPK_TIMEOUT_SEC", "1.0") or "1.0")
                try:
                    spk_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            spk.identify,
                            wav_bytes,
                            16000,
                            1,
                            now_ts=time.time(),
                        ),
                        timeout=timeout_s,
                    )
                except asyncio.TimeoutError:
                    spk_result = None
                except Exception as e:
                    spk_result = None
                    log.info(
                        "ARIA.SPK.Error",
                        extra={"fields": {"session_id": session_id, "utterance": utterance_gen, "error": repr(e)}},
                    )

                if spk_result is not None:
                    speaker_user = spk_result.user
                    log.info(
                        "ARIA.SPK",
                        extra={
                            "fields": {
                                "session_id": session_id,
                                "utterance": utterance_gen,
                                "name": spk_result.name,
                                "score": spk_result.score,
                                "known": spk_result.known,
                                "user": spk_result.user,
                                "second_best": spk_result.second_best,
                            }
                        },
                    )
                    # Suppress if speaker is aria to prevent feedback loop
                    if speaker_user == "aria":
                        should_enqueue = False
                        log.info(
                            "ARIA.SPK: suppressed_aria_speaker",
                            extra={
                                "fields": {
                                    "session_id": session_id,
                                    "utterance": utterance_gen,
                                    "name": spk_result.name,
                                    "score": spk_result.score,
                                    "user": speaker_user,
                                }
                            },
                        )
                    else:
                        # Suppress if self speech (existing logic)
                        self_names = {n.lower() for n in getattr(spk, "self_names", ())}
                        name_lc = spk_result.name.lower()
                        if (
                            spk_result.known
                            and name_lc in self_names
                            and spk_result.score >= getattr(spk, "self_min_score", 0.0)
                        ):
                            should_enqueue = False
                            log.info(
                                "ARIA.SPK: suppressed_self_speech",
                                extra={
                                    "fields": {
                                        "session_id": session_id,
                                        "utterance": utterance_gen,
                                        "name": spk_result.name,
                                        "score": spk_result.score,
                                    }
                                },
                            )

            if should_enqueue:
                echo_guard = getattr(app.state, "echo_guard", None)
                if echo_guard is not None and echo_guard.enabled:
                    suppress, info = echo_guard.should_suppress(text_out, time.time())
                    info = {**info, "session_id": session_id, "utterance": utterance_gen}
                    if suppress:
                        should_enqueue = False
                        log.info("ARIA.ECHO_V2: suppressed", extra={"fields": info})
                    else:
                        log.info("ARIA.ECHO_V2: allowed", extra={"fields": info})

            if should_enqueue:
                print(f"ARIA.TRANSCRIPT: {text_out}", flush=True)
                _enqueue_final_llm(app, text_out, session_id, speaker_user)

        if not client_vad_enabled:
            vad.reset()

        log.info(
            "ARIA.Utterance.Final",
            extra={
                "fields": {
                    "session_id": session_id,
                    "utterance": utterance_gen,
                    "text_len": len(text_out),
                    "bytes": len(wav_bytes),
                    "reason": reason,
                    "mode": "client_vad" if client_vad_enabled else "server_vad",
                    **asr_metrics,
                    "asr_ran": bool(should_asr),
                }
            },
        )

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                if client_vad_enabled and buffering:
                    await _finalize_utterance(reason="disconnect")
                history_lock = getattr(app.state, "llm_history_lock", None)
                history_map = getattr(app.state, "llm_history", None)
                if history_lock is not None and isinstance(history_map, dict):
                    async with history_lock:
                        history_map.pop(session_id, None)
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
                    client_vad_enabled = bool(payload.get("client_vad"))
                    client_vad_started = False
                    log.info(
                        "ARIA.Session.Started",
                        extra={
                            "fields": {
                                "session_id": session_id,
                                "chunk_ms": payload.get("chunk_ms"),
                                "client_vad": client_vad_enabled,
                            }
                        },
                    )
                    continue

                if msg_type == "stop":
                    if client_vad_enabled and buffering:
                        await _finalize_utterance(reason="stop")
                    log.info("ARIA.Session.Stopped", extra={"fields": {"session_id": session_id}})
                    await ws.close()
                    return

                if client_vad_enabled and msg_type == "utterance_start":
                    if buffering:
                        await _finalize_utterance(reason="client_new_start")
                    client_vad_started = True
                    await _start_utterance()
                    continue

                if client_vad_enabled and msg_type == "utterance_end":
                    await _finalize_utterance(reason="client_end")
                    client_vad_started = False
                    continue

                continue

            if data is not None:
                if not started:
                    await ws.close(code=1003)
                    return

                if not client_vad_enabled:
                    update = vad.push(data)
                    if update.utterance_started:
                        await _start_utterance()
                else:
                    # Client-side VAD: only buffer inside explicit utterance boundaries.
                    if not client_vad_started and not buffering:
                        dropped_prestart_bytes += len(data)
                        if dropped_prestart_bytes >= 32000:  # ~1s at 16kHz mono PCM16
                            log.info(
                                "ARIA.ClientVAD.DroppingAudioBeforeStart",
                                extra={
                                    "fields": {
                                        "session_id": session_id,
                                        "dropped_bytes": dropped_prestart_bytes,
                                    }
                                },
                            )
                            dropped_prestart_bytes = 0
                        continue

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

                if not client_vad_enabled and update.utterance_ended and buffering:
                    await _finalize_utterance(reason="server_vad")

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

    should_asr, asr_metrics = _should_run_asr(wav_bytes)
    if not should_asr:
        return
    try:
        text_out = await asyncio.to_thread(asr.transcribe, wav_bytes)
    except Exception as e:
        log.info(
            "ARIA.ASR.PartialError",
            extra={"fields": {"session_id": session_id, "utterance": utterance, "error": repr(e)}},
        )
        return

    text_out = _filter_transcript(
        (text_out or "").strip(),
        seconds=float(asr_metrics.get("seconds", 0.0)),
        speech_ratio=float(asr_metrics.get("speech_ratio", 0.0)),
    )

    # Avoid printing the same partial repeatedly.
    if not text_out or text_out == partial_state.get("last_text", ""):
        return
    partial_state["last_text"] = text_out
    print(f"ARIA(partial) {text_out}", flush=True)


def _llm_history_max_messages() -> int:
    v = os.environ.get("ARIA_LLM_HISTORY_MAX_MESSAGES", "8")
    try:
        max_messages = int(v)
    except ValueError:
        max_messages = 8
    return max(0, max_messages)


def _enqueue_final_llm(app: FastAPI, transcript: str, session_id: str | None, speaker_user: str) -> None:
    """Enqueue a final transcript for LLM processing without blocking."""

    q: asyncio.Queue[tuple[str | None, str, str]] | None = getattr(app.state, "llm_queue", None)
    if q is None:
        return

    text = transcript.strip()
    if not text:
        return

    try:
        q.put_nowait((session_id, text, speaker_user))
    except asyncio.QueueFull:
        # Drop the previous queued item and keep the latest.
        with contextlib.suppress(Exception):
            _ = q.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            q.put_nowait((session_id, text, speaker_user))


async def _llm_worker(app: FastAPI) -> None:
    """Background worker that processes FINAL transcripts via local LLM.

    Runs independently of the WebSocket receive loop so LLM latency cannot block audio ingestion.
    """

    q: asyncio.Queue[tuple[str | None, str, str]] | None = getattr(app.state, "llm_queue", None)
    config = getattr(app.state, "llm_config", None)
    client = getattr(app.state, "llm_client", None)
    if q is None or config is None:
        return

    llm_debug = os.environ.get("ARIA_LLM_DEBUG", "").strip() not in {"", "0", "false", "False"}

    log.info(
        "ARIA.LLM.WorkerStarted",
        extra={"fields": {"url": getattr(config, "url", None), "model": getattr(config, "model", None)}},
    )

    while True:
        item = await q.get()
        if isinstance(item, tuple) and len(item) == 3:
            session_id, transcript, speaker_user = item
        elif isinstance(item, tuple) and len(item) == 2:
            session_id, transcript = item
            speaker_user = "unknown"
        else:
            session_id, transcript, speaker_user = None, str(item), "unknown"
        try:
            # Always emit a low-noise marker so it's obvious when a request is fired.
            # (Full prompt logging is gated behind ARIA_LLM_DEBUG.)
            log.info(
                "ARIA.LLM.Start",
                extra={
                    "fields": {
                        "transcript_len": len(transcript),
                        "queue_pending": q.qsize(),
                    }
                },
            )
            if llm_debug:
                max_chars = int(os.environ.get("ARIA_LLM_DEBUG_MAX_CHARS", "800"))
                user_line = f"[speaker={speaker_user}] {transcript}"
                user_text = user_line if len(user_line) <= max_chars else (user_line[:max_chars] + "…")
                sys_text = SYSTEM_PROMPT
                sys_text = sys_text if len(sys_text) <= 300 else (sys_text[:300] + "…")
                log.info(
                    "ARIA.LLM.Request",
                    extra={
                        "fields": {
                            "system": sys_text,
                            "user": user_text,
                            "transcript_len": len(transcript),
                        }
                    },
                )
            history_max = _llm_history_max_messages()
            messages: list[dict[str, str]] | None = None
            if history_max > 0 and session_id:
                history_lock = getattr(app.state, "llm_history_lock", None)
                history_map = getattr(app.state, "llm_history", None)
                if history_lock is not None and isinstance(history_map, dict):
                    async with history_lock:
                        history = history_map.get(session_id)
                        if history:
                            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *list(history)]

            if messages is None:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                ]
            user_content = f"[speaker={speaker_user}] {transcript}"
            messages.append({"role": "user", "content": user_content})

            t0 = time.monotonic()
            llm_result = await generate_llm_response(
                text=transcript,
                messages=messages,
                config=config,
                client=client,
            )

            # Handle both old (str) and new (dict) return formats
            if isinstance(llm_result, dict):
                response = llm_result.get("content", "")
            else:
                response = llm_result if isinstance(llm_result, str) else ""

            dt_ms = int((time.monotonic() - t0) * 1000)
            if response:
                print(f"ARIA.LLM: {response}", flush=True)
                log.info(
                    "ARIA.LLM.Done",
                    extra={
                        "fields": {
                            "ms": dt_ms,
                            "transcript_len": len(transcript),
                            "response_len": len(response),
                        }
                    },
                )

                if history_max > 0 and session_id:
                    history_lock = getattr(app.state, "llm_history_lock", None)
                    history_map = getattr(app.state, "llm_history", None)
                    if history_lock is not None and isinstance(history_map, dict):
                        async with history_lock:
                            history = history_map.get(session_id)
                            if history is None or getattr(history, "maxlen", None) != history_max:
                                history = deque(list(history or []), maxlen=history_max)
                                history_map[session_id] = history
                            history.append({"role": "user", "content": user_content})
                            history.append({"role": "assistant", "content": response})

                # Optional TTS: speak the LLM response to Sonos.
                speech = getattr(app.state, "speech_output", None)
                if speech is not None:
                    try:
                        await speech.speak(response)
                    except Exception as e:
                        log.info("ARIA.TTS.Error", extra={"fields": {"error": repr(e)}})
            else:
                log.info(
                    "ARIA.LLM.EmptyResponse",
                    extra={"fields": {"ms": dt_ms, "transcript_len": len(transcript)}},
                )
        except Exception as e:
            log.info(
                "ARIA.LLM.Error",
                extra={"fields": {"error": repr(e), "transcript_len": len(transcript)}},
            )


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
