from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import re
import wave
import time
from uuid import uuid4
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.aria_logging import get_logger
from app.audio_sink.sonos_http_sink import (
    SonosDiscoveryError,
    SonosHttpSink,
    discover_sonos_ip,
    verify_sonos_reachable,
)
from app.speech_output.echo_guard_v2 import EchoGuardV2
from app.tts.audio_normalizer import normalize_for_sonos
from app.tts.text_chunker import ChunkingConfig, chunk_text
from app.tts.tts_engine import TtsConfig, TtsEngine, build_tts_engine, load_tts_config


log = get_logger("ARIA")


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _parse_voice_map(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in (value or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        lang, voice = item.split(":", 1)
        lang = lang.strip().lower()
        voice = voice.strip()
        if not lang or not voice:
            continue
        mapping[lang] = voice
    return mapping


def _detect_lang_heuristic(text: str) -> str:
    text = (text or "").lower()
    if not text:
        return "en"
    if re.search(r"[àâäçéèêëîïôöùûüÿœæ]", text):
        return "fr"

    fr_markers = {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "est",
        "sont",
        "être",
        "avoir",
        "je",
        "tu",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "elles",
        "avec",
        "pour",
        "sur",
        "dans",
        "mais",
        "donc",
        "bonjour",
        "merci",
        "salut",
        "mon",
        "ton",
        "nom",
        "comment",
        "quel",
        "quelle",
        "quels",
        "quelles",
        "qui",
        "quoi",
        "s'il",
        "ça",
    }
    en_markers = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "be",
        "have",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "with",
        "for",
        "on",
        "in",
        "but",
        "so",
    }

    tokens = re.findall(r"[a-zàâäçéèêëîïôöùûüÿœæ']+", text)
    fr_score = sum(1 for t in tokens if t in fr_markers)
    en_score = sum(1 for t in tokens if t in en_markers)
    return "fr" if fr_score > en_score else "en"


def _detect_lang_langid(text: str) -> str:
    try:
        import langid
    except Exception:
        return _detect_lang_heuristic(text)
    try:
        lang, _score = langid.classify(text or "")
    except Exception:
        return _detect_lang_heuristic(text)
    return "fr" if lang == "fr" else "en"


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
    echo_guard: EchoGuardV2 | None = field(init=False, default=None, repr=False)
    _engine: TtsEngine | None = field(init=False, default=None, repr=False)
    _sink: SonosHttpSink | None = field(init=False, default=None, repr=False)
    _base_url: str = field(init=False, default="", repr=False)
    _volume_default: float = field(init=False, default=0.3, repr=False)
    _chunking_enabled: bool = field(init=False, default=True, repr=False)
    _max_chars: int = field(init=False, default=220, repr=False)
    _min_chars: int = field(init=False, default=60, repr=False)
    _task: Optional[asyncio.Task[None]] = field(init=False, default=None, repr=False)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _voice_map: dict[str, str] = field(init=False, default_factory=dict, repr=False)
    _lang_detect: str = field(init=False, default="heuristic", repr=False)
    _short_reuse_chars: int = field(init=False, default=20, repr=False)
    _last_lang: str | None = field(init=False, default=None, repr=False)
    _sonos_volume: int | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._engine = build_tts_engine(self.cfg)

        sink = (os.environ.get("ARIA_TTS_SINK", "sonos_http").strip() or "sonos_http").lower()
        if sink != "sonos_http":
            raise ValueError(f"Unsupported ARIA_TTS_SINK={sink!r}")

        discovery_timeout_raw = os.environ.get("ARIA_SONOS_DISCOVERY_TIMEOUT", "5.0") or "5.0"
        try:
            discovery_timeout = float(discovery_timeout_raw)
        except ValueError:
            discovery_timeout = 5.0

        sonos_name = os.environ.get("ARIA_SONOS_NAME", "").strip() or None

        sonos_ip = os.environ.get("ARIA_SONOS_IP", "").strip()
        if not sonos_ip:
            try:
                sonos_ip = discover_sonos_ip(preferred_name=sonos_name, timeout=discovery_timeout)
            except SonosDiscoveryError as exc:
                raise ValueError(str(exc)) from exc

        try:
            speaker_name = verify_sonos_reachable(sonos_ip, timeout=discovery_timeout)
        except SonosDiscoveryError as exc:
            raise ValueError(str(exc)) from exc

        self._sink = SonosHttpSink(speaker_ip=sonos_ip)
        log.info(
            "ARIA.TTS.SonosConfigured",
            extra={"fields": {"ip": sonos_ip, "name": speaker_name or (sonos_name or "<unknown>")}},
        )

        base_url = os.environ.get("ARIA_HTTP_BASE_URL", "").strip().rstrip("/")
        if not base_url:
            raise ValueError("ARIA_HTTP_BASE_URL must be set (e.g. http://192.0.2.10:8000)")
        self._base_url = base_url

        self._volume_default = float(os.environ.get("ARIA_TTS_VOLUME_DEFAULT", "0.3") or "0.3")
        self._chunking_enabled = _truthy_env("ARIA_TTS_CHUNKING", "1")
        self._max_chars = int(os.environ.get("ARIA_TTS_MAX_CHARS_PER_CHUNK", "220") or "220")
        self._min_chars = int(os.environ.get("ARIA_TTS_MIN_CHUNK_CHARS", "60") or "60")

        voice_map_raw = os.environ.get("ARIA_TTS_VOICE_MAP", "").strip()
        if voice_map_raw:
            self._voice_map = _parse_voice_map(voice_map_raw)

        self._lang_detect = (os.environ.get("ARIA_TTS_LANG_DETECT", "heuristic") or "heuristic").strip().lower()
        try:
            self._short_reuse_chars = int(os.environ.get("ARIA_TTS_LANG_SHORT_REUSE_CHARS", "20") or "20")
        except ValueError:
            self._short_reuse_chars = 20

        # Sonos speaker volume (0-100, default 30)
        sonos_vol_raw = os.environ.get("ARIA_SONOS_VOLUME", "").strip()
        if sonos_vol_raw:
            try:
                self._sonos_volume = max(0, min(100, int(sonos_vol_raw)))
            except ValueError:
                self._sonos_volume = None
        else:
            self._sonos_volume = 30  # Default 30%

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

        if self.echo_guard is not None:
            self.echo_guard.set_speaking(True)

        try:
            cfg = ChunkingConfig(
                enabled=self._chunking_enabled,
                max_chars_per_chunk=self._max_chars,
                min_chunk_chars=self._min_chars,
            )
            chunks = chunk_text(text, cfg)
            if not chunks:
                return

            # Set Sonos volume before speaking
            if self._sonos_volume is not None and self._sink is not None:
                try:
                    await self._sink.set_volume(self._sonos_volume)
                except Exception as e:
                    log.warning("ARIA.TTS.VolumeError", extra={"fields": {"error": repr(e)}})

            log.info("ARIA.TTS.Speaking", extra={"fields": {"chunks": len(chunks), "sonos_volume": self._sonos_volume}})

            for chunk in chunks:
                try:
                    voice = self.cfg.voice
                    if self._voice_map:
                        chunk_body = (chunk or "").strip()
                        if len(chunk_body) < self._short_reuse_chars and self._last_lang is not None:
                            lang = self._last_lang
                        else:
                            if self._lang_detect == "langid":
                                lang = _detect_lang_langid(chunk_body)
                            else:
                                lang = _detect_lang_heuristic(chunk_body)
                            self._last_lang = lang
                        voice = self._voice_map.get(lang) or self._voice_map.get("en") or voice
                        log.info(
                            "ARIA.TTS.Language",
                            extra={"fields": {"lang": lang, "voice": voice, "chars": len(chunk_body)}},
                        )

                    key = _hash_key(text=chunk, voice=voice, rate=self.cfg.rate, volume=volume)
                    norm_path = (self.cfg.cache_dir / "sonos" / f"{key}.wav").resolve()

                    if not norm_path.exists() or norm_path.stat().st_size == 0:
                        raw = await asyncio.to_thread(
                            self._engine.synthesize_to_wav, chunk, voice=voice, rate=self.cfg.rate
                        )
                        await asyncio.to_thread(normalize_for_sonos, raw, norm_path, volume=volume)

                    # Record TTS chunk timing before playback (estimated).
                    if self.echo_guard is not None:
                        words = len((chunk or "").strip().split())
                        est_dur = max(0.6, (words / 2.6)) + 0.4
                        start_ts = time.time()
                        end_ts = start_ts + est_dur
                        tts_id = str(uuid4())
                        self.echo_guard.record_tts_event(chunk, start_ts, end_ts, tts_id)

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
                    log.warning(
                        "ARIA.TTS.Error",
                        extra={"fields": {"error": repr(e)}},
                        exc_info=True,
                    )
                    return
        finally:
            if self.echo_guard is not None:
                self.echo_guard.set_speaking(False)


def maybe_build_speech_output(echo_guard: EchoGuardV2 | None = None) -> SpeechOutput | None:
    cfg = load_tts_config()
    if not cfg.enabled:
        return None
    # Ensure cache dir exists
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    speech = SpeechOutput(cfg=cfg)
    speech.echo_guard = echo_guard
    return speech
