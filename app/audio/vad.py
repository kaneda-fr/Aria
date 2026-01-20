"""Voice activity detection.

ARIA uses WebRTC VAD for low-latency speech/silence decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import webrtcvad


@dataclass(frozen=True, slots=True)
class VadConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    aggressiveness: int = 2
    silence_ms: int = 700  # silence threshold for utterance finalization


@dataclass(frozen=True, slots=True)
class VadUpdate:
    in_utterance: bool
    utterance_started: bool
    utterance_ended: bool


class VoiceActivityDetector:
    """Stateful utterance detector based on WebRTC VAD.

    Feed PCM16LE mono bytes; it returns whether we're in an utterance, and whether
    an utterance started/ended at this update.
    """

    def __init__(self, config: VadConfig | None = None) -> None:
        self._cfg = config or VadConfig()
        if self._cfg.sample_rate != 16000:
            raise ValueError("WebRTC VAD config must use 16000 Hz for ARIA")
        if self._cfg.frame_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD supports 10/20/30 ms frames")
        if not (0 <= self._cfg.aggressiveness <= 3):
            raise ValueError("aggressiveness must be 0..3")
        if self._cfg.silence_ms < 0:
            raise ValueError("silence_ms must be >= 0")

        self._vad = webrtcvad.Vad(self._cfg.aggressiveness)
        self._frame_bytes = int(self._cfg.sample_rate * 2 * self._cfg.frame_ms / 1000)
        self._pending = bytearray()

        self._in_utterance = False
        self._silence_ms = 0

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes

    def reset(self) -> None:
        self._pending.clear()
        self._in_utterance = False
        self._silence_ms = 0

    def push(self, pcm16le: bytes) -> VadUpdate:
        """Process audio and update utterance state.

        This method consumes enough audio to make state transitions. It does not
        return per-frame details; callers should buffer audio separately.
        """

        if not pcm16le:
            return VadUpdate(self._in_utterance, False, False)

        self._pending.extend(pcm16le)
        utterance_started = False
        utterance_ended = False

        while len(self._pending) >= self._frame_bytes:
            frame = bytes(self._pending[: self._frame_bytes])
            del self._pending[: self._frame_bytes]

            is_speech = self._vad.is_speech(frame, self._cfg.sample_rate)

            if is_speech:
                if not self._in_utterance:
                    self._in_utterance = True
                    utterance_started = True
                self._silence_ms = 0
            else:
                if self._in_utterance:
                    self._silence_ms += self._cfg.frame_ms
                    if self._silence_ms >= self._cfg.silence_ms:
                        self._in_utterance = False
                        self._silence_ms = 0
                        utterance_ended = True
                        # Keep processing remaining frames for possible next utterance.

        return VadUpdate(self._in_utterance, utterance_started, utterance_ended)
