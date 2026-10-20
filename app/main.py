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

from fastapi import FastAPI, WebSocket, Body, HTTPException
from pydantic import StrictBool

from app.asr.parakeet_onnx import ParakeetOnnx
from app.aria_logging import get_logger
from app.audio.chunking import tail_window
from app.audio.ringbuffer import RingBuffer
from app.audio.vad import VadConfig, VoiceActivityDetector
from app.http_audio.server import build_tts_router
from app.llm.ollama_client import SYSTEM_PROMPT, generate_llm_response, llm_enabled, load_llm_config
from app.llm.request_queue import AssistantRequest, RequestQueue
from app.llm.conversation import ConversationGate, conversation_turn
from app.plugins.registry import PluginRegistry
from app.plugins.inventory import InventoryProvider
from app.plugins.actions import ActionsProvider
from app.routing import get_router
from app.speaker_recognition.recognizer import SpeakerRecognizer
from app.speech_output.echo_guard_v2 import EchoGuardV2
from app.speech_output.speech_output import detect_language, maybe_build_speech_output


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
    app.state.conversation_gate = ConversationGate.from_env()
    app.state.speech_output = None
    app.state.tts_cache_dir = None
    app.state.echo_guard = EchoGuardV2()
    app.state.spk_recognizer = SpeakerRecognizer()
    try:
        if llm_enabled():
            import httpx

            app.state.llm_config = load_llm_config()
            app.state.llm_client = httpx.AsyncClient(timeout=app.state.llm_config.timeout_s)
            from app.llm.ollama_template import prepare_tool_template
            app.state.llm_config = await prepare_tool_template(app.state.llm_config, app.state.llm_client)
            # Bounded queue prevents unbounded task accumulation; enqueue is non-blocking.
            app.state.llm_queue = RequestQueue(
                capacity=int(os.environ.get("ARIA_REQUEST_QUEUE_CAPACITY", "1")),
                max_wait_s=float(os.environ.get("ARIA_REQUEST_MAX_WAIT_S", "30")),
                on_event=_request_event,
            )
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
        if app.state.llm_client is not None:
            await app.state.llm_client.aclose()
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

    # Self-Learning Router (optional, enabled via env)
    app.state.router = None
    try:
        router_enabled = os.environ.get("ARIA_ROUTER_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
        if router_enabled:
            # Per-language embedding models
            model_name_en = os.environ.get("ARIA_ROUTER_MODEL_EN", "BAAI/bge-small-en-v1.5")
            model_name_fr = os.environ.get("ARIA_ROUTER_MODEL_FR", "intfloat/multilingual-e5-small")
            control_threshold = float(os.environ.get("ARIA_ROUTER_CONTROL_THRESHOLD", "0.70"))
            chat_threshold = float(os.environ.get("ARIA_ROUTER_CHAT_THRESHOLD", "0.70"))
            unknown_threshold = float(os.environ.get("ARIA_ROUTER_UNKNOWN_THRESHOLD", "0.50"))
            use_seed_data = os.environ.get("ARIA_ROUTER_USE_SEED_DATA", "1").strip().lower() in {"1", "true", "yes", "on"}
            lang_detect_method = os.environ.get("ARIA_ROUTER_LANG_DETECT", "heuristic")

            app.state.router = get_router(
                model_name_en=model_name_en,
                model_name_fr=model_name_fr,
                control_threshold=control_threshold,
                chat_threshold=chat_threshold,
                unknown_threshold=unknown_threshold,
                use_seed_data=use_seed_data,
                lang_detect_method=lang_detect_method
            )

            router_stats = app.state.router.get_stats()
            log.info(
                "ARIA.Router.Enabled",
                extra={
                    "fields": {
                        "model_en": model_name_en,
                        "model_fr": model_name_fr,
                        "control_threshold": control_threshold,
                        "chat_threshold": chat_threshold,
                        "unknown_threshold": unknown_threshold,
                        "lang_detect_method": lang_detect_method,
                        "centroid_distance_en": router_stats.get("centroid_distance_en"),
                        "centroid_distance_fr": router_stats.get("centroid_distance_fr"),
                        "centroids_initialized": router_stats.get("centroids_initialized")
                    }
                }
            )
        else:
            log.info("ARIA.Router.Disabled", extra={"fields": {}})
    except Exception as e:
        log.error("ARIA.Router.InitError", extra={"fields": {"error": repr(e)}})
        app.state.router = None

    yield

    # Stop request execution before the providers it may be using.
    queue = getattr(app.state, "llm_queue", None)
    if queue is not None:
        queue.closing = True
    worker = getattr(app.state, "llm_worker_task", None)
    if worker is not None:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    plugin_registry = getattr(app.state, "plugin_registry", None)
    if plugin_registry is not None:
        try:
            log.info("ARIA.Plugins.Stopping")
            plugin_registry.stop_all()
            log.info("ARIA.Plugins.Stopped")
        except Exception as e:
            log.error("ARIA.Plugins.StopError", extra={"fields": {"error": repr(e)}})

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
    """Admit text to the same bounded worker used by final voice transcripts.

    Existing fields remain; replies add request_id/status. Full queues return 429;
    expired pending work returns 504 without execution.
    """
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    text = request.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "Missing or invalid 'text' parameter")
    session_id = request.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise HTTPException(400, "session_id must be a string")
    if session_id is not None and len(session_id) > 256:
        raise HTTPException(400, "session_id must be at most 256 characters")
    speaker_user = request.get("speaker_user", "unknown")
    if not isinstance(speaker_user, str):
        raise HTTPException(400, "speaker_user must be a string")
    for key in ("speak_response", "use_tools"):
        if key in request and not isinstance(request[key], bool):
            raise HTTPException(400, f"{key} must be a boolean")
    queue = getattr(app.state, "llm_queue", None)
    if queue is None:
        raise HTTPException(503, "LLM not enabled")
    job = queue.submit(text=text.strip(), session_id=session_id, speaker_user=speaker_user,
                       speak_response=request.get("speak_response", False),
                       use_tools=request.get("use_tools", True), source="api")
    try:
        result = await asyncio.shield(job.result)
    except asyncio.CancelledError:
        queue.cancel_pending(job)
        raise
    error = result.get("error")
    status_code = {"busy": 429, "shutdown": 503, "queue_expired": 504,
                   "caller_cancelled": 409, "execution_interrupted": 503}.get(error)
    if status_code is None and result["status"] == "failed":
        status_code = 500
    if status_code is not None:
        return JSONResponse(result, status_code=status_code,
                            headers={"Retry-After": "1"} if error == "busy" else None)
    return result


@conversation_turn
async def _process_request(app: FastAPI, job: AssistantRequest) -> dict[str, Any]:
    """The single conversation/tool/history/speech path for both transports."""
    from app.llm.control import ControlMemory, Plan, resolve, render
    from app.llm.local_answers import LocalAnswer, clock_answer, capability_answer
    from app.llm.lookup_reply import render_lookup_results
    from app.plugins.inventory.read_requests import resolve_read_request, answer_read_request
    from app.llm.history_prefill import cancel_prefill, schedule_prefill

    await cancel_prefill(app.state)
    from fastapi import HTTPException
    from app.llm.tools import available_tools, execute_tool, format_tool_result_for_llm
    from app.plugins.inventory.state_read import render_state, measurement_request, resolved_read_request, READ_WORDS
    from app.plugins.inventory.selection import location_needs_clarification
    from app.plugins.tooling_provider import ToolingProvider

    text = job.conversation_text if job.conversation_text is not None else job.text
    session_id, speaker_user = job.session_id, job.speaker_user
    speak_response, use_tools = job.speak_response, job.use_tools
    language = detect_language(text)
    log.info("ARIA.LLM.Start", extra={"fields": {
        "request_id": job.request_id, "session_id": session_id, "source": job.source, "language": language,
        "transcript_len": len(text), "queue_pending": app.state.llm_queue.pending.qsize()}})
    if os.environ.get("ARIA_LLM_DEBUG", "").strip().lower() not in {"", "0", "false"}:
        max_chars = max(0, int(os.environ.get("ARIA_LLM_DEBUG_MAX_CHARS", "800")))
        user_line = f"[speaker={speaker_user}] {text}"
        log.info("ARIA.LLM.Request", extra={"fields": {
            "request_id": job.request_id, "source": job.source, "system": SYSTEM_PROMPT[:300],
            "user": user_line[:max_chars], "transcript_len": len(text)}})
    from app.llm.client_mute import MuteCommands
    mute_commands = getattr(app.state, 'mute_commands', None)
    if mute_commands is None:
        mute_commands = app.state.mute_commands = MuteCommands()
    mute_reply = mute_commands.handle(text, session_id, speaker_user,
                                      getattr(app.state, 'client_sessions', None))
    if mute_reply is not None:
        log.info('ARIA.Session.MuteCommand', extra={'fields': {
            'request_id': job.request_id, 'session_id': session_id,
            'client_id': mute_reply.client_id, 'status': mute_reply.status,
            'duration_s': mute_reply.seconds}})
        spoken = await _speak_reply(app, mute_reply.text, mute_reply.language) if speak_response else False
        return {'success': True, 'text': text, 'response': mute_reply.text,
                'error': None, 'spoken': spoken, 'tool_calls_made': 0,
                'mute_status': mute_reply.status}
    shortcut = _chat_shortcut(app, text)
    if shortcut is not None:
        spoken = await _speak_reply(app, shortcut, language) if speak_response else False
        return {"success": True, "text": text, "response": shortcut, "error": None,
                "spoken": spoken, "tool_calls_made": 0}

    # Check if LLM is enabled
    llm_config = getattr(app.state, "llm_config", None)
    llm_client = getattr(app.state, "llm_client", None)

    # Get plugin providers for tools
    plugin_registry = getattr(app.state, "plugin_registry", None)
    inventory_provider = None
    actions_provider = None

    if use_tools and plugin_registry:
        providers = plugin_registry.get_plugins_by_type(ToolingProvider)
        inventory_provider = next((p for p in providers if p.name == "inventory"), None)
        actions_provider = plugin_registry.get_plugin("actions")

    refresh_inventory = getattr(inventory_provider, 'refresh_live_inventory', None)
    if refresh_inventory is not None:
        await refresh_inventory()
    request_tools = available_tools(inventory_provider) if use_tools and inventory_provider is not None else None

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
        read_history = list(messages[1:-1])

        # Tool calling loop
        tool_calls_made = 0
        max_iterations = 5
        t0 = time.monotonic()
        final_response = ""
        iteration = -1
        control_plan = None
        read_plan = None
        action_result = None
        local_answer = clock_answer(text) or capability_answer(
            text, inventory_provider, actions_provider, tools_enabled=use_tools)
        if (local_answer is None and use_tools and callable(getattr(inventory_provider, 'read_state', None))
                and getattr(inventory_provider, 'indexer', None) is not None
                and measurement_request(text)
                and location_needs_clarification(inventory_provider.indexer.get_snapshot(), text, extra=READ_WORDS)):
            local_answer = LocalAnswer(render_state({'status': 'ambiguous'}, language),
                                       language, 'read_clarification')
        if (local_answer is None and use_tools and callable(getattr(inventory_provider, 'read_state', None))
                and getattr(inventory_provider, 'indexer', None) is not None):
            read_args = resolved_read_request(inventory_provider.indexer.get_snapshot(), text, read_history)
            if read_args is not None:
                read_result = await execute_tool(
                    tool_name='get_device_state',
                    tool_args={**read_args, 'request_text': text, 'request_history': read_history},
                    inventory_provider=inventory_provider, actions_provider=actions_provider)
                tool_calls_made = 1
                local_answer = LocalAnswer(render_state(read_result, language), language, 'read_clarification')
                memory = getattr(app.state, 'control_memory', None)
                if memory is None:
                    memory = app.state.control_memory = ControlMemory()
                snapshot = inventory_provider.indexer.get_snapshot()
                selected = {d.eq_id for d in snapshot.devices.values()
                            for c in d.commands
                            if any(row.get('cmd_id') == c.cmd_id
                                   for row in read_result.get('readings', []))}
                if len(selected) == 1:
                    memory.remember((session_id, speaker_user), next(iter(selected)))
                else:
                    memory.forget((session_id, speaker_user))
        if local_answer is not None:
            final_response, language = local_answer.text, local_answer.language
            log.info("ARIA.Local.Answer", extra={"fields": {
                "request_id": job.request_id, "kind": local_answer.kind}})
        if use_tools and local_answer is None:
            memory = getattr(app.state, "control_memory", None)
            if memory is None:
                memory = app.state.control_memory = ControlMemory()
            reference_key = (session_id, speaker_user)
            indexer = getattr(inventory_provider, 'indexer', None)
            snapshot = (indexer.get_snapshot() if indexer is not None and
                        getattr(inventory_provider, 'is_started', False) else None)
            read_plan = resolve_read_request(text, snapshot, memory.get(reference_key))
            if read_plan is not None:
                language = read_plan.language or language
                final_response, read_result = await answer_read_request(read_plan, inventory_provider, language)
                tool_calls_made = int(read_plan.status == 'ready')
                log.info('ARIA.Inventory.Lookup', extra={'fields': {
                    'request_id': job.request_id, 'session_id': session_id,
                    'kind': read_plan.kind, 'status': read_plan.status,
                    'device_id': read_plan.device_id, 'metric': read_plan.metric,
                    'match': read_plan.match, 'result': read_result, 'language': language, 'model_calls': 0}})
                if read_plan.status == 'ready' and read_result.get('status') != 'not_found':
                    memory.remember(reference_key, read_plan.device_id)
                else:
                    memory.forget(reference_key)
            else:
                control_plan = resolve(text, inventory_provider, memory.get(reference_key))
            if control_plan is not None:
                language = control_plan.language
                log.info("ARIA.Control.Resolved", extra={"fields": {
                    "request_id": job.request_id, "status": control_plan.status,
                    "command": control_plan.command}})
                if control_plan.status == "ready":
                    grant = control_plan.authorization
                    tool_calls_made = 1
                    action_result = await execute_tool(
                        tool_name="execute_command",
                        tool_args={"cmd_id": grant.cmd_id, **({"value": grant.value} if grant.value is not None else {})},
                        inventory_provider=inventory_provider, actions_provider=actions_provider,
                        authorization=grant)
                    if action_result.get("success"):
                        memory.remember(reference_key, grant.device_id)
                    else:
                        memory.forget(reference_key)
                else:
                    memory.forget(reference_key)
                final_response = render(control_plan, action_result)

        handled = control_plan is not None or local_answer is not None or read_plan is not None
        if not handled and (not llm_config or not llm_client):
            raise HTTPException(status_code=503, detail="LLM not enabled")
        for iteration in range(0 if handled else max_iterations):
            # Generate LLM response with tools
            llm_result = await generate_llm_response(
                text=text,
                messages=messages,
                config=llm_config,
                client=llm_client,
                tools=request_tools,
            )

            app.state.llm_recent_session = session_id
            app.state.llm_recent_at = time.monotonic()
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

            # A read-only search batch is grounded evidence, not another generation task.
            lookup_results = []
            lookup_only = all(call.get('function', {}).get('name') in ('find_devices', 'get_device_state')
                              for call in tool_calls)
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

                if (tool_name == 'find_devices' and callable(getattr(inventory_provider, 'read_state', None))
                        and measurement_request(text)):
                    tool_name = 'get_device_state'
                    log.info('ARIA.Control.ReadIntent', extra={'fields': {
                        'request_id': job.request_id, 'original_tool': 'find_devices',
                        'tool': tool_name}})
                if tool_name in ('find_devices', 'get_device_state'):
                    tool_args = {**tool_args, 'request_text': text, 'request_history': read_history}
                # Execute tool
                tool_result = await execute_tool(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    inventory_provider=inventory_provider,
                    actions_provider=actions_provider
                )
                if tool_name == "execute_command":
                    # No server-grounded action exists on this conversation path.
                    # Do not let another generated reply invent execution success.
                    final_response = render(Plan("unsupported", language))
                    break

                if lookup_only:
                    lookup_results.append((tool_name, tool_result))
                    continue

                # Format result for LLM
                tool_result_text = format_tool_result_for_llm(tool_result, language=language)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": tool_result_text,
                    "name": tool_name,
                    "tool_name": tool_name,
                })
            if lookup_only:
                searches = [result for name, result in lookup_results if name == 'find_devices']
                readings = [result for name, result in lookup_results if name == 'get_device_state']
                parts = ([render_lookup_results(searches, language=language)] if searches else [])
                parts.extend(render_state(result, language) for result in readings)
                final_response = ' '.join(parts)
                log.info('ARIA.Lookup.Reply', extra={'fields': {
                    'request_id': job.request_id, 'searches': len(lookup_results),
                    'language': language, 'model_round_trip_saved': True}})
            if final_response:
                break

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
                extra={"fields": {"ms": dt_ms, "transcript_len": len(text), "source": job.source, "request_id": job.request_id, "tool_calls_made": tool_calls_made}}
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
                    "source": job.source, "request_id": job.request_id
                }
            },
        )

        prefill_messages = None
        history_evicted = False
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
                    history_evicted = len(history) + 2 > history_max
                    history.append({"role": "user", "content": user_content})
                    history.append({"role": "assistant", "content": final_response})
                    prefill_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *list(history)]

        # Optional TTS
        spoken = await _speak_reply(app, final_response, language) if speak_response else False
        read_cache = (read_plan is not None or (local_answer is not None and local_answer.kind == 'read_clarification')) and llm_config is not None and llm_client is not None
        if (read_cache or (read_plan is None and (local_answer is not None or history_evicted))) and spoken and prefill_messages is not None:
            schedule_prefill(app.state, session_id=session_id, messages=prefill_messages,
                             tools=request_tools,
                             audio_ready=getattr(app.state.speech_output, "audio_ready", None),
                             request_id=job.request_id, rollover=history_evicted, server_read=read_cache)

        return {
            "success": True,
            "text": text,
            "response": final_response,
            "error": None,
            "spoken": spoken,
            "tool_calls_made": tool_calls_made,
            "control_status": control_plan.status if control_plan else None,
            "lookup_status": read_plan.status if read_plan else None,
            "action_outcome": action_result.get("outcome") if action_result else None,
        }

    except Exception as e:
        log.error(
            "ARIA.LLM.Error",
            extra={"fields": {"error": repr(e), "transcript_len": len(text), "source": job.source, "request_id": job.request_id}},
        )
        raise


@app.get("/api/clients")
async def client_status() -> dict:
    registry = getattr(app.state, "client_sessions", None)
    return {"clients": registry.snapshot() if registry else []}


@app.post("/api/clients/clear-offline")
async def clear_offline_clients() -> dict:
    registry = getattr(app.state, "client_sessions", None)
    cleared = registry.clear_offline() if registry else []
    log.info("ARIA.Session.ClearOffline", extra={"fields": {"cleared_count": len(cleared)}})
    return {"cleared": cleared, "clients": registry.snapshot() if registry else []}


@app.post("/api/clients/{client_id}/mute")
async def mute_client(client_id: str, muted: StrictBool = Body(embed=True, strict=True)) -> dict:
    registry = getattr(app.state, "client_sessions", None)
    if registry is None or client_id not in registry.known:
        raise HTTPException(404, "Unknown client")
    commands = getattr(app.state, 'mute_commands', None)
    if commands is not None:
        commands.forget_session('client:' + client_id)
    status = registry.set_muted(client_id, muted)
    log.info("ARIA.Session.Mute", extra={"fields": {
        "client_id": client_id, "session_id": "client:" + client_id, **status}})
    return {"client_id": client_id, **status}


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket) -> None:
    from app.client_sessions import ClientSessions, validate_client_id
    session_id = uuid4().hex
    connection_id = session_id
    client_id = None
    client_sessions = getattr(app.state, "client_sessions", None)
    if client_sessions is None:
        client_sessions = app.state.client_sessions = ClientSessions()
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
    utterance_started_at: float | None = None
    partial_task: asyncio.Task[None] | None = None
    mute_revision = 0

    status_lock = asyncio.Lock()

    async def _send_status(message: dict) -> None:
        try:
            async with status_lock:
                identity = {"client_id": client_id, "connection_id": connection_id} if client_id else {}
                await asyncio.wait_for(ws.send_json({**message, **identity}), timeout=.2)
        except Exception:
            pass  # Terminal state is also recorded in structured server logs.

    async def _start_utterance() -> None:
        nonlocal buffering, utterance_gen, last_partial_at, utterance_started_at
        utterance_gen += 1
        audio.clear()
        buffering = True
        last_partial_at = time.monotonic()
        utterance_started_at = last_partial_at
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

        revision = client_sessions.mute_revision(client_id)
        if client_sessions.is_muted(client_id) or revision != mute_revision:
            return

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

        if client_sessions.is_muted(client_id) or client_sessions.mute_revision(client_id) != revision:
            return

        # Drop common filler hallucinations for low-speech utterances.
        text_out = _filter_transcript(
            text_out,
            seconds=float(asr_metrics.get("seconds", 0.0)),
            speech_ratio=float(asr_metrics.get("speech_ratio", 0.0)),
        )

        # Spec change: server prints transcripts; nothing is sent back to clients.

        should_enqueue = True
        speaker_user = "unknown"
        spk_result = None
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
                    # Nearest identity is independent of confidence. Only a
                    # confident self match may reject speech; keep raw scores above.
                    from app.speaker_recognition.recognizer import self_speech_decision
                    speaker_user, suppress_self = self_speech_decision(
                        spk_result, getattr(spk, "self_names", ()),
                        getattr(spk, "self_min_score", 0.75))
                    if suppress_self:
                        should_enqueue = False
                    if spk_result.user.lower() == "aria" or suppress_self:
                        log.info(
                            "ARIA.SPK: suppressed_self_speech" if suppress_self else
                            "ARIA.SPK: uncertain_self_match",
                            extra={"fields": {
                                "session_id": session_id, "utterance": utterance_gen,
                                "name": spk_result.name, "score": spk_result.score,
                                "known": spk_result.known, "user": speaker_user,
                                "self_min_score": getattr(spk, "self_min_score", 0.75),
                            }},
                        )

            if should_enqueue:
                echo_guard = getattr(app.state, "echo_guard", None)
                if echo_guard is not None and echo_guard.enabled:
                    echo_now = time.time()
                    onset_ts = echo_now - max(0.0, time.monotonic() - utterance_started_at)
                    suppress, info = echo_guard.should_suppress(
                        text_out, echo_now, speaker_user, onset_ts)
                    info = {**info, "session_id": session_id, "utterance": utterance_gen,
                            "client_id": client_id, "speaker_user": speaker_user,
                            "speaker_score": spk_result.score if spk_result is not None else None}
                    if suppress:
                        should_enqueue = False
                        log.info("ARIA.ECHO_V2: suppressed", extra={"fields": info})
                    else:
                        log.info("ARIA.ECHO_V2: allowed", extra={"fields": info})

            if client_sessions.is_muted(client_id) or client_sessions.mute_revision(client_id) != revision:
                return

            if should_enqueue:
                print(f"ARIA.TRANSCRIPT: {text_out}", flush=True)
                job = _enqueue_final_llm(app, text_out, session_id, speaker_user,
                                         notify=_send_status, started_at=utterance_started_at)
                if job is None:
                    await _send_status({"type": "request_status", "session_id": session_id,
                                        "status": "rejected", "error": "unavailable"})
                else:
                    log.info("ARIA.SPK.Request", extra={"fields": {
                        "request_id": job.request_id, "session_id": session_id,
                        "client_id": client_id, "utterance": utterance_gen,
                        "user": speaker_user,
                        "name": spk_result.name if spk_result is not None else None,
                        "score": spk_result.score if spk_result is not None else None,
                        "known": bool(spk_result is not None and spk_result.known)}})
                    await _send_status(job.status_message(
                        "accepted" if job.status == "queued" else job.status,
                        job.result.result().get("error") if job.result.done() else None))

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
            revision = client_sessions.mute_revision(client_id)
            muted = client_sessions.is_muted(client_id)
            if revision != mute_revision or muted:
                # A mute transition invalidates the entire unfinished utterance,
                # including a mute/unmute pair between two received frames.
                audio.clear()
                buffering = client_vad_started = False
                vad.reset()
                if partial_task is not None:
                    partial_task.cancel()
                    partial_task = None
                mute_revision = revision
            if message.get("type") == "websocket.disconnect":
                if client_vad_enabled and buffering:
                    await _finalize_utterance(reason="disconnect")
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

                    try:
                        name = validate_client_id(payload)
                        if name is not None:
                            session_id = client_sessions.claim(name, connection_id)
                            client_id = name
                            mute_revision = client_sessions.mute_revision(client_id)
                    except ValueError as exc:
                        await _send_status({"type": "session_error", "error": str(exc)})
                        await ws.close(code=1008)
                        return
                    started = True
                    client_vad_enabled = bool(payload.get("client_vad"))
                    client_vad_started = False
                    log.info(
                        "ARIA.Session.Started",
                        extra={
                            "fields": {
                                "session_id": session_id,
                                "connection_id": connection_id,
                                "client_id": client_id,
                                "chunk_ms": payload.get("chunk_ms"),
                                "client_vad": client_vad_enabled,
                            }
                        },
                    )
                    if client_id:
                        await _send_status({"type": "session_started", "session_id": session_id})
                    continue

                if msg_type == "stop":
                    if client_vad_enabled and buffering:
                        await _finalize_utterance(reason="stop")
                    log.info("ARIA.Session.Stopped", extra={"fields": {"session_id": session_id}})
                    await ws.close()
                    return

                if muted:
                    continue

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
                if data:
                    client_sessions.audio(client_id, connection_id)
                if muted:
                    continue

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
        client_sessions.release(client_id, connection_id)
        # Named clients retain bounded conversation state across transport loss.
        # The conversation gate still clears it on goodbye, expiry or eviction.
        if client_id is None:
            history_lock = getattr(app.state, "llm_history_lock", None)
            history_map = getattr(app.state, "llm_history", None)
            if history_lock is not None and isinstance(history_map, dict):
                async with history_lock:
                    history_map.pop(session_id, None)
        log.info("ARIA.Session.Disconnected", extra={"fields": {
            "session_id": session_id, "connection_id": connection_id, "client_id": client_id}})


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


def _request_event(fields: dict) -> None:
    registry = getattr(app.state, "client_sessions", None)
    if registry is not None:
        registry.request(fields)
    log.info("ARIA.Request.Status", extra={"fields": fields})


def _enqueue_final_llm(app: FastAPI, transcript: str, session_id: str | None,
                       speaker_user: str, notify=None, *, started_at: float | None = None) -> AssistantRequest | None:
    """Non-blocking voice admission; never displace another accepted request."""
    queue = getattr(app.state, "llm_queue", None)
    if queue is None or not transcript.strip():
        _request_event({"session_id": session_id, "request_id": uuid4().hex,
                        "source": "voice", "status": "rejected", "error": "unavailable"})
        return None
    return queue.submit(text=transcript.strip(), session_id=session_id, speaker_user=speaker_user,
                        speak_response=True, use_tools=True, source="voice", notify=notify,
                        conversation_started_at=started_at)


async def _llm_worker(app: FastAPI) -> None:
    from app.llm.history_prefill import cancel_prefill
    try:
        await app.state.llm_queue.run(lambda job: _process_request(app, job))
    finally:
        await cancel_prefill(app.state)






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


def _chat_shortcut(app: FastAPI, text: str) -> str | None:
    """Only explicit phrases are shortcuts; CHAT is not a canned-answer category."""
    router = getattr(app.state, 'router', None)
    if router is None:
        return None
    try:
        decision = router.classify(text)
        log.info('ARIA.Router.Decision', extra={'fields': {
            'mode': decision.mode, 'detected_lang': decision.detected_lang,
            'confidence': round(decision.confidence, 3)}})
        if decision.mode != 'chat':
            return None
        responses = {
            'bonjour': 'Bonjour! Comment puis-je vous aider?', 'bonsoir': 'Bonsoir!',
            'salut': 'Salut!', 'bonne nuit': 'Bonne nuit!', 'coucou': 'Coucou!',
            'hello': 'Hello! How can I help you?', 'hi': 'Hi there!', 'hey': 'Hey!',
            'good morning': 'Good morning!', 'good evening': 'Good evening!',
            'good night': 'Good night!', 'merci': 'De rien!',
            'merci beaucoup': 'Je vous en prie!', 'thank you': "You're welcome!",
            'thanks': "You're welcome!", 'au revoir': 'Au revoir!',
            'goodbye': 'Goodbye!', 'à bientôt': 'À bientôt!',
        }
        key = text.lower().strip().rstrip('?!.,')
        return responses.get(key)
    except Exception as exc:
        log.info('ARIA.Router.ClassifyError', extra={'fields': {'error': repr(exc)}})
        return None


async def _speak_reply(app: FastAPI, text: str, language: str) -> bool:
    speech = getattr(app.state, 'speech_output', None)
    if speech is not None:
        try:
            scheduled = await speech.speak(text, language=language)
            return scheduled is not False
        except Exception as exc:
            log.info('ARIA.TTS.Error', extra={'fields': {'error': repr(exc)}})
    return False
