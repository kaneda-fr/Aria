from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


try:  # Optional dependency
    from rapidfuzz import fuzz as _rf_fuzz
except Exception:  # pragma: no cover - optional
    _rf_fuzz = None


_STOPWORDS = {
    # English
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
    # French
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
    "etre",
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
}


@dataclass(frozen=True, slots=True)
class EchoV2Config:
    enabled: bool
    tail_sec: float
    min_words: int
    min_chars: int
    containment_threshold: float
    fuzzy_threshold: float
    allowlist_regex: str | None
    unknown_prefix_tail_sec: float = 8.0

    @staticmethod
    def from_env() -> "EchoV2Config":
        return EchoV2Config(
            unknown_prefix_tail_sec=max(0.0, min(30.0, _env_float("ARIA_ECHO_V2_UNKNOWN_PREFIX_TAIL_SEC", 8.0))),
            enabled=_truthy_env("ARIA_ECHO_V2_ENABLED", "1"),
            tail_sec=_env_float("ARIA_ECHO_V2_TAIL_SEC", 2.0),
            min_words=_env_int("ARIA_ECHO_V2_MIN_WORDS", 5),
            min_chars=_env_int("ARIA_ECHO_V2_MIN_CHARS", 20),
            containment_threshold=_env_float("ARIA_ECHO_V2_CONTAINMENT_THRESHOLD", 0.65),
            fuzzy_threshold=_env_float("ARIA_ECHO_V2_FUZZY_THRESHOLD", 0.75),
            allowlist_regex=os.environ.get("ARIA_ECHO_V2_ALLOWLIST_REGEX", "").strip() or None,
        )


@dataclass(frozen=True, slots=True)
class TtsEvent:
    tts_id: str
    text: str
    start_ts: float
    end_ts_est: float


class EchoGuardV2:
    def __init__(self, cfg: EchoV2Config | None = None) -> None:
        self._cfg = cfg or EchoV2Config.from_env()
        self._lock = threading.Lock()
        self._events: list[TtsEvent] = []
        self._speaking = False

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def record_tts_event(self, text: str, start_ts: float, end_ts_est: float, tts_id: str) -> None:
        if not text.strip():
            return
        with self._lock:
            self._events.append(TtsEvent(tts_id=tts_id, text=text, start_ts=start_ts, end_ts_est=end_ts_est))
            if len(self._events) > 2:
                self._events = self._events[-2:]

    def set_speaking(self, is_speaking: bool) -> None:
        with self._lock:
            self._speaking = bool(is_speaking)

    def is_speaking(self, now_ts: float) -> bool:
        with self._lock:
            if not self._speaking:
                return False
            last = self._events[-1] if self._events else None
            if last is None:
                return self._speaking
            if now_ts > (last.end_ts_est + self._cfg.tail_sec):
                self._speaking = False
            return self._speaking

    def should_suppress(self, transcript: str, now_ts: float,
                        speaker_user: str | None = None,
                        utterance_started_ts: float | None = None) -> tuple[bool, dict[str, Any]]:
        info: dict[str, Any] = {
            "reason": None,
            "speaker_user": speaker_user,
            "prefix_words": 0,
            "containment": 0.0,
            "fuzzy": 0.0,
            "matched_event": None,
            "tts_id": None,
            "transcript_len": len(transcript or ""),
            "candidate_len": 0,
        }

        if not self._cfg.enabled:
            return False, info

        text = (transcript or "").strip()
        if not text:
            return False, info

        with self._lock:
            events = list(self._events)
            speaking = self._speaking

        # The capture onset, not delayed ASR completion, determines echo proximity.
        capture_ts = now_ts if utterance_started_ts is None else utterance_started_ts
        if not (now_ts - 60.0 <= capture_ts <= now_ts):
            return False, info
        info["capture_age_s"] = now_ts - capture_ts

        if self._cfg.allowlist_regex:
            try:
                if re.search(self._cfg.allowlist_regex, text, flags=re.IGNORECASE):
                    return False, info
            except re.error:
                pass

        if speaker_user == "unknown":
            for event in reversed(events):
                if not (event.start_ts <= capture_ts <= event.end_ts_est + self._cfg.unknown_prefix_tail_sec):
                    continue
                prefix_words = _matching_opening(text, event.text)
                if prefix_words:
                    info.update(reason="unknown_speaker_prefix", prefix_words=prefix_words,
                                tts_id=event.tts_id, candidate_len=len(event.text),
                                matched_event="current" if event is events[-1] else "last")
                    return True, info

        # Existing whole-text evidence remains independent of speaker attribution.
        events = [event for event in events if event.start_ts <= capture_ts <=
                  event.end_ts_est + self._cfg.tail_sec]
        if not events:
            return False, info

        norm_t = _normalize(text)
        tokens_t = _tokens(norm_t)
        tokens_t = [t for t in tokens_t if len(t) >= 2 and t not in _STOPWORDS]
        if len(tokens_t) < self._cfg.min_words or len(norm_t) < self._cfg.min_chars:
            return False, info

        candidates: list[tuple[str, str]] = []
        if len(events) >= 1:
            candidates.append(("current", events[-1].text))
        if len(events) >= 2:
            candidates.append(("last", events[-2].text))

        for label, cand_text in candidates:
            norm_s = _normalize(cand_text)
            tokens_s = _tokens(norm_s)
            tokens_s = [t for t in tokens_s if len(t) >= 2 and t not in _STOPWORDS]
            if not tokens_s:
                continue

            containment = _containment(tokens_t, tokens_s)
            fuzzy = _fuzzy_score(tokens_t, tokens_s)

            if containment >= self._cfg.containment_threshold and fuzzy >= self._cfg.fuzzy_threshold:
                matched = events[-1] if label == "current" else events[-2]
                info.update(
                    {
                        "reason": "text_similarity",
                        "containment": containment,
                        "fuzzy": fuzzy,
                        "matched_event": label,
                        "tts_id": matched.tts_id,
                        "candidate_len": len(cand_text),
                    }
                )
                return True, info

            # Keep best scores for logging even if not suppressed.
            if containment > info["containment"] or fuzzy > info["fuzzy"]:
                info.update(
                    {
                        "containment": max(info["containment"], containment),
                        "fuzzy": max(info["fuzzy"], fuzzy),
                        "matched_event": label,
                        "tts_id": (events[-1] if label == "current" else events[-2]).tts_id,
                        "candidate_len": len(cand_text),
                    }
                )

        return False, info


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _matching_opening(transcript: str, spoken: str) -> int:
    """Exact normalized opening; numeric short replies need a numeric continuation."""
    incoming, reference = _normalize(transcript).split(), _normalize(spoken).split()
    count = min(3, len(reference))
    if count < 2 or incoming[:count] != reference[:count]:
        return 0
    if count == 2:
        if not re.search(r'\d', spoken):
            return 0
        # Digits in TTS become words in ASR. Do not match an unrelated continuation
        # such as "It is possible ..." merely because a clock reply began "It is".
        words = re.findall(r'\w+', _strip_accents(transcript.lower()))
        numbers = set(('zero un une deux trois quatre cinq six sept huit neuf dix onze '
                       'douze treize quatorze quinze seize vingt trente quarante cinquante '
                       'soixante septante huitante nonante cent mille one two three four '
                       'five six seven eight nine ten eleven twelve thirteen fourteen '
                       'fifteen sixteen seventeen eighteen nineteen twenty thirty forty '
                       'fifty sixty seventy eighty ninety hundred thousand').split())
        if len(words) < 3 or not (words[2].isdigit() or words[2] in numbers):
            return 0
    return count


def _normalize(text: str) -> str:
    text = text.lower()
    text = _strip_accents(text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _tokens(text: str) -> list[str]:
    return [t for t in text.split() if t]


def _containment(tokens_t: list[str], tokens_s: list[str]) -> float:
    if not tokens_t:
        return 0.0
    set_s = set(tokens_s)
    match = sum(1 for t in tokens_t if t in set_s)
    return match / float(len(tokens_t))


def _fuzzy_score(tokens_t: list[str], tokens_s: list[str]) -> float:
    if not tokens_t or not tokens_s:
        return 0.0
    if _rf_fuzz is not None:
        t = " ".join(sorted(set(tokens_t)))
        s = " ".join(sorted(set(tokens_s)))
        return float(_rf_fuzz.token_set_ratio(t, s)) / 100.0

    t = " ".join(sorted(set(tokens_t)))
    s = " ".join(sorted(set(tokens_s)))
    return SequenceMatcher(None, t, s).ratio()


def now_ts() -> float:
    return time.time()
