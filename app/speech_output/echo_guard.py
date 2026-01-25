from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher


_INTERRUPT_KEYWORDS = {"stop", "cancel", "aria"}


def _truthy_env(name: str, default: str = "0") -> bool:
    import os

    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    import os

    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    import os

    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_set(text: str) -> set[str]:
    if not text:
        return set()
    return set(text.split())


@dataclass(frozen=True, slots=True)
class EchoGuardConfig:
    enabled: bool
    window_sec: float
    min_chars: int
    min_words: int
    seq_threshold: float
    jacc_threshold: float
    partial_threshold: float
    overlap_threshold: float
    speaking_grace_sec: float
    strict_while_speaking: bool

    @staticmethod
    def from_env() -> "EchoGuardConfig":
        return EchoGuardConfig(
            enabled=_truthy_env("ARIA_ECHO_GUARD_ENABLED", "1"),
            window_sec=_env_float("ARIA_ECHO_GUARD_WINDOW_SEC", 15.0),
            min_chars=_env_int("ARIA_ECHO_GUARD_MIN_CHARS", 25),
            min_words=_env_int("ARIA_ECHO_GUARD_MIN_WORDS", 6),
            seq_threshold=_env_float("ARIA_ECHO_GUARD_SEQ_THRESHOLD", 0.85),
            jacc_threshold=_env_float("ARIA_ECHO_GUARD_JACC_THRESHOLD", 0.70),
            partial_threshold=_env_float("ARIA_ECHO_GUARD_PARTIAL_THRESHOLD", 0.78),
            overlap_threshold=_env_float("ARIA_ECHO_GUARD_OVERLAP_THRESHOLD", 0.60),
            speaking_grace_sec=_env_float("ARIA_ECHO_GUARD_SPEAKING_GRACE_SEC", 6.0),
            strict_while_speaking=_truthy_env("ARIA_ECHO_GUARD_STRICT_WHILE_SPEAKING", "1"),
        )


@dataclass(frozen=True, slots=True)
class TtsChunk:
    text: str
    start_ts: float
    end_ts: float


class EchoGuard:
    """Text-level echo suppression for Sonos playback feedback.

    Thread-safe/async-safe via an internal lock.
    """

    def __init__(self, cfg: EchoGuardConfig | None = None) -> None:
        self._cfg = cfg or EchoGuardConfig.from_env()
        self._lock = threading.Lock()
        self._chunks: list[TtsChunk] = []
        self._speaking = False

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def set_speaking(self, speaking: bool) -> None:
        with self._lock:
            self._speaking = bool(speaking)

    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking

    def record_tts_chunk(self, text: str, start_ts: float, end_ts: float) -> None:
        if not text.strip():
            return
        with self._lock:
            self._chunks.append(TtsChunk(text=text, start_ts=float(start_ts), end_ts=float(end_ts)))
            self._prune_locked(now_ts=end_ts)

    def should_suppress(self, transcript: str, now_ts: float, speaking: bool) -> bool:
        if not self._cfg.enabled:
            return False
        if not transcript or not transcript.strip():
            return False

        norm = _normalize(transcript)
        if not norm:
            return False

        words = norm.split()
        # Do not suppress interrupts.
        if _is_interrupt(words):
            return False

        with self._lock:
            self._prune_locked(now_ts=now_ts)
            if not self._chunks:
                return False

            # Only suppress when speaking or recently spoken (~1s).
            last_end = self._chunks[-1].end_ts
            within_grace = speaking or (now_ts - last_end) <= self._cfg.speaking_grace_sec
            if not within_grace:
                return False

            recent_texts = [c.text for c in self._chunks]

        if self._cfg.strict_while_speaking:
            return True

        # Suppress very short utterances while speaking/recently spoken.
        if len(norm) < self._cfg.min_chars or len(words) < self._cfg.min_words:
            return True

        # Compare against each recent chunk and concatenations of recent chunks.
        candidates: list[str] = []
        candidates.extend(recent_texts)
        if len(recent_texts) >= 2:
            candidates.append(" ".join(recent_texts[-3:]))
        if len(recent_texts) >= 4:
            candidates.append(" ".join(recent_texts[-6:]))

        for cand in candidates:
            cand_norm = _normalize(cand)
            if not cand_norm:
                continue

            seq_sim = SequenceMatcher(None, norm, cand_norm).ratio()
            jacc = _jaccard(norm, cand_norm)
            overlap = _overlap_ratio(norm, cand_norm)

            if (
                seq_sim >= self._cfg.seq_threshold
                or jacc >= self._cfg.jacc_threshold
                or overlap >= self._cfg.overlap_threshold
            ):
                return True

            # Partial suppression for longer transcripts and substring matches.
            if len(norm) >= self._cfg.min_chars:
                if norm in cand_norm or cand_norm in norm:
                    return True
                if seq_sim >= self._cfg.partial_threshold:
                    return True

        return False

    def _prune_locked(self, now_ts: float) -> None:
        window = self._cfg.window_sec
        if window <= 0:
            self._chunks.clear()
            return
        cutoff = now_ts - window
        self._chunks = [c for c in self._chunks if c.end_ts >= cutoff]


def _jaccard(a: str, b: str) -> float:
    sa = _word_set(a)
    sb = _word_set(b)
    if not sa or not sb:
        return 0.0
    inter = sa.intersection(sb)
    union = sa.union(sb)
    return len(inter) / len(union)


def _overlap_ratio(a: str, b: str) -> float:
    sa = _word_set(a)
    sb = _word_set(b)
    if not sa or not sb:
        return 0.0
    inter = sa.intersection(sb)
    denom = min(len(sa), len(sb))
    if denom <= 0:
        return 0.0
    return len(inter) / denom


def _is_interrupt(words: list[str]) -> bool:
    if not words:
        return False
    word_set = set(words)
    if {"stop", "cancel"}.intersection(word_set):
        return True
    if "aria" in word_set and len(words) <= 4:
        return True
    return False


def now_ts() -> float:
    return time.monotonic()
