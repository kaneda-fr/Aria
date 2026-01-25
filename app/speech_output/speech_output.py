from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.aria_logging import get_logger
from app.audio_sink.sonos_http_sink import SonosHttpSink
from app.tts.audio_normalizer import normalize_for_sonos
from app.tts.text_chunker import ChunkingConfig, chunk_text
from app.tts.tts_engine import TtsConfig, TtsEngine, build_tts_engine, load_tts_config


log = get_logger("ARIA")


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _hash_key(*, text: str, voice: str, rate: float, volume: float) -> str:
    h = hashlib.sha256()
    h.update(text.strip().encode("utf-8"))
    h.update(b"\n")
    h.update(voice.strip().encode("utf-8"))
    h.update(b"\n")
    h.update(f"{rate:.4f}".encode("ascii"))
    h.update(b"\n")
    h.update(f"{volume:.3f}".encode("ascii"))
    return h.hexdigest()[:32]


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate() or 44100
        return float(frames) / float(rate)


@dataclass(slots=True)
class SpeechOutput:
    cfg: TtsConfig
    _engine: TtsEngine | None = field(init=False, default=None, repr=False)
    _sink: SonosHttpSink | None = field(init=False, default=None, repr=False)
    _base_url: str = field(init=False, default="", repr=False)
    _volume_default: float = field(init=False, default=0.3, repr=False)
    _chunking_enabled: bool = field(init=False, default=True, repr=False)
    _max_chars: int = field(init=False, default=220, repr=False)
    _min_chars: int = field(init=False, default=60, repr=False)
    _task: Optional[asyncio.Task[None]] = field(init=False, default=None, repr=False)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self._engine = build_tts_engine(self.cfg)

        sink = (os.environ.get("ARIA_TTS_SINK", "sonos_http").strip() or "sonos_http").lower()
        if sink != "sonos_http":
            raise ValueError(f"Unsupported ARIA_TTS_SINK={sink!r}")

        sonos_ip = os.environ.get("ARIA_SONOS_IP", "").strip()
        if not sonos_ip:
            raise ValueError("ARIA_SONOS_IP must be set when ARIA_TTS_SINK=sonos_http")

        self._sink = SonosHttpSink(speaker_ip=sonos_ip)

        base_url = os.environ.get("ARIA_HTTP_BASE_URL", "").strip().rstrip("/")
        if not base_url:
            raise ValueError("ARIA_HTTP_BASE_URL must be set (e.g. http://192.168.100.100:8000)")
        self._base_url = base_url

        self._volume_default = float(os.environ.get("ARIA_TTS_VOLUME_DEFAULT", "0.3") or "0.3")
        self._chunking_enabled = _truthy_env("ARIA_TTS_CHUNKING", "1")
        self._max_chars = int(os.environ.get("ARIA_TTS_MAX_CHARS_PER_CHUNK", "220") or "220")
        self._min_chars = int(os.environ.get("ARIA_TTS_MIN_CHUNK_CHARS", "60") or "60")

        # _task and _lock are initialized by dataclass defaults.

    def tts_cache_dir(self) -> Path:
        return self.cfg.cache_dir

    async def speak(self, text: str, volume: float | None = None) -> None:
        """Start speaking `text`.

        Non-blocking: this schedules a background task.
        """

        # Cancel any existing playback first (outside the lock to avoid deadlocks).
        prior: Optional[asyncio.Task[None]]
        async with self._lock:
            prior = self._task
            self._task = None

        if prior is not None and not prior.done():
            prior.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await prior
            with contextlib.suppress(Exception):
                await self._sink.stop_all()

        vol = self._volume_default if volume is None else float(volume)
        async with self._lock:
            self._task = asyncio.create_task(self._run_speak(text, vol))

    async def interrupt(self) -> None:
        """Stop current playback and cancel queued chunks.

        NOTE: Per user request, we DO NOT automatically call this from VAD yet.
        """

        async with self._lock:
            t = self._task
            self._task = None

        if t is not None and not t.done():
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t

        with contextlib.suppress(Exception):
            await self._sink.stop_all()

    async def _run_speak(self, text: str, volume: float) -> None:
        text = (text or "").strip()
        if not text:
            return

        cfg = ChunkingConfig(
            enabled=self._chunking_enabled,
            max_chars_per_chunk=self._max_chars,
            min_chunk_chars=self._min_chars,
        )
        chunks = chunk_text(text, cfg)
        if not chunks:
            return

        log.info("ARIA.TTS.Speaking", extra={"fields": {"chunks": len(chunks)}})

        for chunk in chunks:
            try:
                key = _hash_key(text=chunk, voice=self.cfg.voice, rate=self.cfg.rate, volume=volume)
                norm_path = (self.cfg.cache_dir / "sonos" / f"{key}.wav").resolve()

                if not norm_path.exists() or norm_path.stat().st_size == 0:
                    raw = await asyncio.to_thread(
                        self._engine.synthesize_to_wav, chunk, voice=self.cfg.voice, rate=self.cfg.rate
                    )
                    await asyncio.to_thread(normalize_for_sonos, raw, norm_path, volume=volume)

                url = f"{self._base_url}/tts/{key}.wav"
                log.info("ARIA.TTS.URL", extra={"fields": {"url": url}})

                # Fire playback and then wait approximately the duration.
                await self._sink.play_url(url)
                try:
                    dur = _wav_duration_seconds(norm_path)
                except Exception:
                    dur = 1.0
                await asyncio.sleep(max(0.2, dur + 0.15))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.info("ARIA.TTS.Error", extra={"fields": {"error": repr(e)}})
                return


def maybe_build_speech_output() -> SpeechOutput | None:
    cfg = load_tts_config()
    if not cfg.enabled:
        return None
    # Ensure cache dir exists
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return SpeechOutput(cfg=cfg)
