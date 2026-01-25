from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.aria_logging import get_logger

try:  # Optional heavy deps
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None

try:
    from speechbrain.inference import EncoderClassifier
except Exception:  # pragma: no cover
    EncoderClassifier = None


@dataclass(frozen=True, slots=True)
class SpeakerResult:
    name: str
    score: float
    known: bool
    user: str
    second_best: float


@dataclass(frozen=True, slots=True)
class SpeakerConfig:
    enabled: bool
    profiles_dir: Path
    model: str
    device: str
    num_threads: int
    min_seconds: float
    threshold_default: float
    self_min_score: float
    timeout_sec: float
    hot_reload: bool
    self_names: tuple[str, ...]

    @staticmethod
    def from_env() -> "SpeakerConfig":
        profiles_dir = Path(os.environ.get("ARIA_SPK_PROFILES_DIR", "/var/aria/profiles"))
        model = os.environ.get("ARIA_SPK_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
        device = os.environ.get("ARIA_SPK_DEVICE", "cpu")
        num_threads = _env_int("ARIA_SPK_NUM_THREADS", 4)
        min_seconds = _env_float("ARIA_SPK_MIN_SECONDS", 1.0)
        threshold_default = _env_float("ARIA_SPK_THRESHOLD_DEFAULT", 0.65)
        self_min_score = _env_float("ARIA_SPK_SELF_MIN_SCORE", 0.75)
        timeout_sec = _env_float("ARIA_SPK_TIMEOUT_SEC", 1.0)
        hot_reload = _truthy_env("ARIA_SPK_HOT_RELOAD", "0")
        raw_self = os.environ.get("ARIA_SPK_SELF_NAMES", "")
        self_list = [s.strip().lower() for s in raw_self.split(",") if s.strip()]
        return SpeakerConfig(
            enabled=_truthy_env("ARIA_SPK_ENABLED", "1"),
            profiles_dir=profiles_dir,
            model=model,
            device=device,
            num_threads=num_threads,
            min_seconds=min_seconds,
            threshold_default=threshold_default,
            self_min_score=self_min_score,
            timeout_sec=timeout_sec,
            hot_reload=hot_reload,
            self_names=tuple(self_list),
        )


class SpeakerRecognizer:
    def __init__(self, cfg: SpeakerConfig | None = None) -> None:
        self._cfg = cfg or SpeakerConfig.from_env()
        self._lock = threading.Lock()
        self._profiles: list[tuple[str, np.ndarray]] = []
        self._encoder = None
        self._last_loaded = 0.0
        self._ready = False
        self._log = get_logger("ARIA")
        self._debug = _truthy_env("ARIA_SPK_DEBUG", "0")

        if not self._cfg.enabled:
            self._log.info("ARIA.SPK.Disabled")
            return

        self._load_profiles()
        try:
            self._load_model()
        except Exception as e:
            self._encoder = None
            self._log.info("ARIA.SPK.ModelLoadError", extra={"fields": {"error": repr(e)}})

        self._ready = bool(self._profiles) and self._encoder is not None
        if not self._ready:
            reason = "missing_profiles" if not self._profiles else "missing_model"
            self._log.info("ARIA.SPK.NotReady", extra={"fields": {"reason": reason}})

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def self_names(self) -> tuple[str, ...]:
        return self._cfg.self_names

    @property
    def self_min_score(self) -> float:
        return self._cfg.self_min_score

    def _load_profiles(self) -> None:
        profiles_dir = self._cfg.profiles_dir
        profiles_dir.mkdir(parents=True, exist_ok=True)
        profiles: list[tuple[str, np.ndarray]] = []
        for path in sorted(profiles_dir.glob("*.json")):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                name = str(obj.get("name", "")).strip()
                emb = np.array(obj.get("embedding", []), dtype=np.float32)
                if not name or emb.size == 0:
                    continue
                emb = _l2_normalize_np(emb)
                profiles.append((name, emb))
            except Exception:
                continue
        with self._lock:
            self._profiles = profiles
            self._last_loaded = time.time()
        self._log.info(
            "ARIA.SPK.ProfilesLoaded",
            extra={"fields": {"count": len(profiles), "dir": str(profiles_dir)}},
        )

    def _load_model(self) -> None:
        if EncoderClassifier is None or torch is None:
            self._encoder = None
            self._log.info("ARIA.SPK.ModelUnavailable")
            return
        if self._cfg.num_threads > 0:
            try:
                torch.set_num_threads(int(self._cfg.num_threads))
            except Exception:
                pass
        self._encoder = EncoderClassifier.from_hparams(
            source=self._cfg.model,
            run_opts={"device": self._cfg.device},
        )
        self._log.info(
            "ARIA.SPK.ModelLoaded",
            extra={"fields": {"model": self._cfg.model, "device": self._cfg.device}},
        )
        # Warm-up with short silence
        try:
            dummy = torch.zeros(1, 16000)
            with torch.no_grad():
                _ = self._encoder.encode_batch(dummy)
        except Exception:
            pass

    def _maybe_reload(self) -> None:
        if not self._cfg.hot_reload:
            return
        now = time.time()
        if now - self._last_loaded < 2.0:
            return
        self._load_profiles()

    def identify(self, audio: np.ndarray | bytes, sample_rate: int, channels: int, *, now_ts: float) -> SpeakerResult:
        if not self._cfg.enabled:
            return SpeakerResult("unknown", 0.0, False, "unknown", 0.0)
        self._maybe_reload()

        with self._lock:
            profiles = list(self._profiles)
            encoder = self._encoder

        if not profiles or encoder is None or torch is None or F is None:
            return SpeakerResult("unknown", 0.0, False, "unknown", 0.0)

        wav = _to_float32(audio, sample_rate, channels)
        if wav is None:
            return SpeakerResult("unknown", 0.0, False, "unknown", 0.0)

        if sample_rate != 16000:
            wav = _resample(wav, sample_rate, 16000)
            sample_rate = 16000

        if wav.size == 0:
            return SpeakerResult("unknown", 0.0, False, "unknown", 0.0)

        seconds = wav.shape[-1] / float(sample_rate)
        if seconds < self._cfg.min_seconds:
            return SpeakerResult("unknown", 0.0, False, "unknown", 0.0)

        tensor = torch.from_numpy(wav).unsqueeze(0)
        with torch.no_grad():
            emb = encoder.encode_batch(tensor).squeeze().cpu()
        emb = F.normalize(emb, p=2, dim=0)

        scores: list[tuple[str, float]] = []
        for name, ref in profiles:
            ref_t = torch.from_numpy(ref)
            sim = float(F.cosine_similarity(emb, ref_t, dim=0))
            scores.append((name, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = scores[0]
        second_best = scores[1][1] if len(scores) > 1 else 0.0
        known = best_score >= self._cfg.threshold_default
        user = _base_user(best_name)
        if self._debug:
            top = scores[:5]
            self._log.info(
                "ARIA.SPK.Debug",
                extra={"fields": {"top": [{"name": n, "score": s} for n, s in top]}},
            )
        return SpeakerResult(best_name, best_score, known, user, second_best)


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _to_float32(audio: np.ndarray | bytes, sample_rate: int, channels: int) -> np.ndarray | None:
    if isinstance(audio, bytes):
        if channels <= 0:
            return None
        data = np.frombuffer(audio, dtype=np.int16)
        if data.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        wav = data.astype(np.float32) / 32768.0
        return wav

    if isinstance(audio, np.ndarray):
        wav = audio.astype(np.float32, copy=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav

    return None


def _resample(wav: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return wav
    if torchaudio is not None and torch is not None:
        try:
            t = torch.from_numpy(wav).unsqueeze(0)
            res = torchaudio.functional.resample(t, in_sr, out_sr).squeeze(0).numpy()
            return res
        except Exception:
            pass

    # Fallback: simple linear interpolation
    ratio = float(out_sr) / float(in_sr)
    new_len = int(round(wav.shape[-1] * ratio))
    if new_len <= 1:
        return wav[:1]
    x_old = np.linspace(0.0, 1.0, num=wav.shape[-1], endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def _l2_normalize_np(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x)
    if denom <= 0:
        return x
    return (x / denom).astype(np.float32)


def _base_user(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "unknown"
    lowered = name.lower()
    if lowered.endswith("_en") or lowered.endswith("_fr"):
        return name[:-3]
    return name
