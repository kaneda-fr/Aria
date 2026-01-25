from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TtsConfig:
    enabled: bool
    engine: str
    voice: str
    rate: float
    cache_dir: Path
    timeout_sec: float


class TtsEngine(Protocol):
    def synthesize_to_wav(self, text: str, *, voice: str, rate: float) -> Path:
        """Return a path to a raw WAV file for `text`.

        The returned path must exist when this method returns.
        """


def load_tts_config() -> TtsConfig:
    import os

    enabled = os.environ.get("ARIA_TTS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    engine = os.environ.get("ARIA_TTS_ENGINE", "piper").strip() or "piper"
    voice = os.environ.get("ARIA_TTS_VOICE", "").strip()
    rate_s = os.environ.get("ARIA_TTS_RATE", "1.0").strip() or "1.0"
    cache_dir_s = os.environ.get("ARIA_TTS_CACHE_DIR", "/tmp/aria_tts_cache").strip() or "/tmp/aria_tts_cache"
    timeout_s = os.environ.get("ARIA_TTS_TIMEOUT_SEC", "10").strip() or "10"

    try:
        rate = float(rate_s)
    except ValueError:
        rate = 1.0
    if rate <= 0:
        rate = 1.0

    try:
        timeout_sec = float(timeout_s)
    except ValueError:
        timeout_sec = 10.0
    if timeout_sec <= 0:
        timeout_sec = 10.0

    return TtsConfig(
        enabled=enabled,
        engine=engine,
        voice=voice,
        rate=rate,
        cache_dir=Path(cache_dir_s),
        timeout_sec=timeout_sec,
    )


def build_tts_engine(cfg: TtsConfig) -> TtsEngine:
    engine = (cfg.engine or "").strip().lower()
    if engine == "piper":
        from app.tts.piper_engine import PiperEngine

        return PiperEngine(cache_dir=cfg.cache_dir, timeout_sec=cfg.timeout_sec)

    raise ValueError(f"Unsupported ARIA_TTS_ENGINE={cfg.engine!r}")
