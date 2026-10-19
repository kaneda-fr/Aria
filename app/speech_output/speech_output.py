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
        "combien",
        "pourquoi",
        "quand",
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


def detect_language(text: str) -> str:
    """Select FR/EN locally from this request; ambiguous input defaults to EN.

    No session/global language memory and no model downloads. Optional langid
    retains the existing heuristic fallback when unavailable.
    """
    mode = (os.environ.get("ARIA_TTS_LANG_DETECT", "heuristic") or "heuristic").strip().lower()
    return _detect_lang_langid(text) if mode == "langid" else _detect_lang_heuristic(text)


def speech_language(text: str, requested: str) -> str:
    """Use clear prose evidence; neutral numbers/names keep the request language."""
    words = set(re.findall(r"[a-zàâçéèêîôùûü]+", (text or '').lower()))
    if words and words <= {'oui', 'non', 'bonjour', 'bonsoir', 'merci', 'salut', 'désolé'}:
        return 'fr'
    if words and words <= {'yes', 'no', 'hello', 'thanks', 'sorry', 'goodbye'}:
        return 'en'
    fr = len(words & {'je', 'tu', 'il', 'elle', 'nous', 'vous', 'le', 'la', 'les',
                      'une', 'des', 'du', 'est', 'sont', 'ne', 'pas', 'votre',
                      'pour', 'dans', 'quel', 'quelle', 'peux', 'pouvez'})
    en = len(words & {'i', 'you', 'he', 'she', 'we', 'they', 'the', 'a', 'an',
                      'is', 'are', 'not', 'your', 'could', 'would', 'please',
                      'with', 'for', 'in', 'what', 'which', 'can'})
    if en >= 2 and en - fr >= 2:
        return 'en'
    if fr >= 2 and fr - en >= 2:
        return 'fr'
    return requested


def speech_chunks(text: str, cfg: ChunkingConfig, requested: str) -> list[tuple[str, str]]:
    """Keep existing size bounds, but never merge sentences across voice changes.

    Ambiguous text keeps the last clear language in this reply, initially the
    request's language. Never inherit another request's voice. Same-language chunks retain their original text
    and cache keys. This is voice routing, not a mixed-language safety guard.
    """
    from app.tts.text_chunker import _split_sentences

    result: list[tuple[str, str]] = []
    current = requested
    for chunk in chunk_text(text, cfg):
        sentences = _split_sentences(chunk)
        groups: list[tuple[str, str]] = []
        for sentence in sentences:
            lang = speech_language(sentence, current)
            current = lang
            if groups and groups[-1][1] == lang:
                groups[-1] = (groups[-1][0] + ' ' + sentence, lang)
            else:
                groups.append((sentence, lang))
        if len(groups) == 1:
            groups[0] = (chunk, groups[0][1])
        result.extend(groups)
    return result


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
    audio_ready: asyncio.Future[bool] | None = field(init=False, default=None, repr=False)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _schedule_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _playing: bool = field(init=False, default=False, repr=False)
    _input_revision: int = field(init=False, default=0, repr=False)
    _quiet_until: float = field(init=False, default=0.0, repr=False)
    _voice_map: dict[str, str] = field(init=False, default_factory=dict, repr=False)
    _sonos_volume: int | None = field(init=False, default=None, repr=False)
    _output_mute_deadline: float = field(init=False, default=0.0, repr=False)
    _output_mute_revision: int = field(init=False, default=0, repr=False)

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

    @property
    def input_revision(self) -> int:
        return self._input_revision

    @property
    def input_blocked(self) -> bool:
        """Drain microphones without processing output or its short acoustic tail."""
        return self._playing or time.monotonic() < self._quiet_until

    @property
    def playback_done(self) -> asyncio.Task | None:
        """Snapshot of synthesis and acknowledged (or estimated) sink completion."""
        return self._task

    def output_status(self) -> dict:
        remaining = max(0., self._output_mute_deadline - time.monotonic())
        return {'available': True, 'muted': remaining > 0, 'mute_remaining_s': remaining}

    def set_output_muted(self, muted: bool) -> dict:
        """Skip future clips for five minutes without stopping an active delivery."""
        if self.output_status()['muted'] != muted:
            self._output_mute_deadline = time.monotonic() + 300 if muted else 0.
            self._output_mute_revision += 1
        return self.output_status()

    def _skip_output(self) -> bool:
        self.audio_ready = asyncio.get_running_loop().create_future()
        self.audio_ready.set_result(False)
        log.info('ARIA.TTS.Skipped', extra={'fields': {'reason': 'output_muted'}})
        return False

    async def speak(self, text: str, volume: float | None = None, *, language: str | None = None) -> bool:
        """Schedule one reply after preceding output finishes; no automatic interruption."""
        if language is not None and language not in {'en', 'fr'}:
            raise ValueError('Speech language must be en, fr, or None')
        if self.output_status()['muted']:
            return self._skip_output()
        language = language or detect_language(text)
        mute_revision = self._output_mute_revision
        # The shared worker bounds waiting producers; never accumulate playback tasks.
        async with self._schedule_lock:
            prior = self._task
            if prior is not None and not prior.done():
                try:
                    await asyncio.shield(prior)
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        raise
            if self.output_status()['muted'] or mute_revision != self._output_mute_revision:
                return self._skip_output()
            vol = self._volume_default if volume is None else float(volume)
            async with self._lock:
                self.audio_ready = asyncio.get_running_loop().create_future()
                self._task = asyncio.create_task(self._run_speak(text, vol, language, self.audio_ready, mute_revision))
                ready = self.audio_ready
                self._task.add_done_callback(lambda _: ready.set_result(False) if not ready.done() else None)
        return True

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

    async def _run_speak(self, text: str, volume: float, language: str | None = None,
                         audio_ready: asyncio.Future[bool] | None = None,
                         output_revision: int | None = None) -> None:
        mute_revision = self._output_mute_revision if output_revision is None else output_revision

        def muted() -> bool:
            return self.output_status()['muted'] or mute_revision != self._output_mute_revision

        if muted():
            log.info('ARIA.TTS.Skipped', extra={'fields': {'reason': 'output_muted'}})
            return
        text = (text or "").strip()
        if not text:
            return
        requested = language or detect_language(text)
        pending = None

        async def prepare(chunk: str, voice: str):
            if muted():
                return None
            key = _hash_key(text=chunk, voice=voice, rate=self.cfg.rate, volume=volume)
            norm_path = (self.cfg.cache_dir / "sonos" / f"{key}.wav").resolve()
            if not norm_path.exists() or norm_path.stat().st_size == 0:
                raw = await asyncio.to_thread(
                    self._engine.synthesize_to_wav, chunk, voice=voice, rate=self.cfg.rate
                )
                if muted():
                    return None
                await asyncio.to_thread(normalize_for_sonos, raw, norm_path, volume=volume)
            if muted():
                return None
            try:
                dur = _wav_duration_seconds(norm_path)
            except Exception:
                dur = max(0.6, len(chunk.split()) / 2.6) + 0.4
            return key, dur

        if self.echo_guard is not None:
            self.echo_guard.set_speaking(True)
        try:
            cfg = ChunkingConfig(
                enabled=self._chunking_enabled,
                max_chars_per_chunk=self._max_chars,
                min_chunk_chars=self._min_chars,
            )
            chunks = speech_chunks(text, cfg, requested)
            if not chunks:
                return

            def voice_for(lang: str) -> str:
                return self._voice_map.get(lang) or self._voice_map.get("en") or self.cfg.voice

            # One future WAV, only for replies requiring different voices.
            # Ordinary monolingual output retains its existing scheduling/cache path.
            lookahead = len({voice_for(lang) for _, lang in chunks}) > 1

            if self._sonos_volume is not None and self._sink is not None:
                try:
                    await self._sink.set_volume(self._sonos_volume)
                except Exception as e:
                    log.warning("ARIA.TTS.VolumeError", extra={"fields": {"error": repr(e)}})

            log.info("ARIA.TTS.Speaking", extra={"fields": {
                "chunks": len(chunks), "sonos_volume": self._sonos_volume}})

            for chunk_index, (chunk, lang) in enumerate(chunks):
                if muted():
                    log.info('ARIA.TTS.Skipped', extra={'fields': {'reason': 'output_muted'}})
                    return
                voice = voice_for(lang)
                if lang != requested:
                    log.warning('ARIA.TTS.LanguageMismatch', extra={'fields': {
                        'requested_language': requested, 'speech_language': lang,
                        'chars': len(chunk), 'chunk_index': chunk_index}})
                log.info("ARIA.TTS.Language", extra={"fields": {
                    "lang": lang, "voice": voice, "chars": len(chunk),
                    "chunk_index": chunk_index, "requested_language": requested}})
                if pending is None:
                    prepared = await prepare(chunk, voice)
                else:
                    prepared = await pending
                    pending = None
                if prepared is None or muted():
                    return
                key, dur = prepared

                if self.echo_guard is not None:
                    start_ts = time.time()
                    self.echo_guard.record_tts_event(
                        chunk, start_ts, start_ts + dur, str(uuid4()))

                url = f"{self._base_url}/tts/{key}.wav"
                log.info("ARIA.TTS.URL", extra={"fields": {"url": url}})
                if not self._playing:
                    self._playing = True
                    self._input_revision += 1
                delivery = await self._sink.play_url(url)
                if chunk_index == len(chunks) - 1 and audio_ready is not None and not audio_ready.done():
                    audio_ready.set_result(True)
                if lookahead and chunk_index + 1 < len(chunks) and not muted():
                    next_text, next_lang = chunks[chunk_index + 1]
                    pending = asyncio.create_task(prepare(next_text, voice_for(next_lang)))
                if isinstance(delivery, asyncio.Future):
                    await delivery
                else:
                    await asyncio.sleep(max(0.2, dur + 0.15))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("ARIA.TTS.Error", extra={"fields": {"error": repr(e)}}, exc_info=True)
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            if self._playing:
                self._playing = False
                self._input_revision += 1
                self._quiet_until = time.monotonic() + 0.3
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
