"""Voice activity detection.

ARIA uses WebRTC VAD for low-latency speech/silence decisions.
"""

from __future__ import annotations

import audioop
from dataclasses import dataclass

import math

import webrtcvad


@dataclass(frozen=True, slots=True)
class VadConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    aggressiveness: int = 2
    # Require consecutive speech frames to start an utterance (noise click protection).
    start_frames: int = 3
    # Optional energy gating (disabled when 0.0).
    min_rms: float = 0.0
    # Optional SNR gating in dB (disabled when 0.0).
    snr_db: float = 0.0
    # EMA update rate for noise floor estimate (0..1).
    noise_alpha: float = 0.05
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
        if self._cfg.start_frames < 1:
            raise ValueError("start_frames must be >= 1")
        if self._cfg.min_rms < 0:
            raise ValueError("min_rms must be >= 0")
        if self._cfg.snr_db < 0:
            raise ValueError("snr_db must be >= 0")
        if not (0.0 <= float(self._cfg.noise_alpha) <= 1.0):
            raise ValueError("noise_alpha must be 0..1")
        if self._cfg.silence_ms < 0:
            raise ValueError("silence_ms must be >= 0")

        self._vad = webrtcvad.Vad(self._cfg.aggressiveness)
        self._frame_bytes = int(self._cfg.sample_rate * 2 * self._cfg.frame_ms / 1000)
        self._pending = bytearray()

        self._in_utterance = False
        self._silence_ms = 0
        self._speech_run = 0
        self._noise_rms = 0.0

        self._snr_mult = 1.0
        if self._cfg.snr_db > 0:
            # dB for amplitude ratio: 20*log10(A_signal/A_noise)
            self._snr_mult = math.pow(10.0, self._cfg.snr_db / 20.0)

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes

    def reset(self) -> None:
        self._pending.clear()
        self._in_utterance = False
        self._silence_ms = 0
        self._speech_run = 0
        self._noise_rms = 0.0

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

            # Compute short-term energy and update noise floor (when not in utterance).
            frame_rms = float(audioop.rms(frame, 2)) / 32768.0
            if not self._in_utterance:
                a = float(self._cfg.noise_alpha)
                if self._noise_rms <= 0.0:
                    self._noise_rms = frame_rms
                else:
                    self._noise_rms = (1.0 - a) * self._noise_rms + a * frame_rms

            energy_ok = True
            if self._cfg.min_rms > 0.0 and frame_rms < self._cfg.min_rms:
                energy_ok = False
            if energy_ok and self._cfg.snr_db > 0.0:
                denom = self._noise_rms if self._noise_rms > 1e-6 else 1e-6
                if (frame_rms / denom) < self._snr_mult:
                    energy_ok = False

            is_speech_eff = bool(is_speech) and energy_ok

            if not self._in_utterance:
                if is_speech_eff:
                    self._speech_run += 1
                else:
                    self._speech_run = 0

                if self._speech_run >= self._cfg.start_frames:
                    self._in_utterance = True
                    utterance_started = True
                    self._silence_ms = 0
                    self._speech_run = 0
            else:
                if is_speech_eff:
                    self._silence_ms = 0
                else:
                    self._silence_ms += self._cfg.frame_ms
                    if self._silence_ms >= self._cfg.silence_ms:
                        self._in_utterance = False
                        self._silence_ms = 0
                        utterance_ended = True
                        # Keep processing remaining frames for possible next utterance.

        return VadUpdate(self._in_utterance, utterance_started, utterance_ended)
