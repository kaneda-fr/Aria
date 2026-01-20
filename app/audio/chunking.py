"""Streaming window/chunking logic.

These helpers keep audio framing consistent with ARIA's strict contract.
"""

from __future__ import annotations


def bytes_per_ms(*, sample_rate: int, channels: int, sample_width_bytes: int) -> int:
    if sample_rate <= 0 or channels <= 0 or sample_width_bytes <= 0:
        raise ValueError("Invalid audio format")
    # sample_rate is frames/sec; per ms: sample_rate/1000 frames.
    return int(sample_rate * channels * sample_width_bytes / 1000)


def chunk_bytes(
    data: bytes,
    *,
    sample_rate: int,
    channels: int,
    sample_width_bytes: int = 2,
    chunk_ms: int = 40,
) -> list[bytes]:
    """Split raw PCM bytes into fixed-sized chunks.

    Drops any trailing partial chunk.
    """

    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be > 0")
    bpm = bytes_per_ms(sample_rate=sample_rate, channels=channels, sample_width_bytes=sample_width_bytes)
    chunk_size = bpm * chunk_ms
    if chunk_size <= 0:
        raise ValueError("Invalid chunk size")

    out: list[bytes] = []
    for off in range(0, len(data) - (len(data) % chunk_size), chunk_size):
        out.append(data[off : off + chunk_size])
    return out


def tail_window(
    data: bytes,
    *,
    sample_rate: int,
    channels: int,
    sample_width_bytes: int = 2,
    window_ms: int = 2000,
) -> bytes:
    """Return the last `window_ms` of PCM bytes (or all if shorter)."""

    if window_ms <= 0:
        raise ValueError("window_ms must be > 0")
    bpm = bytes_per_ms(sample_rate=sample_rate, channels=channels, sample_width_bytes=sample_width_bytes)
    keep = bpm * window_ms
    if keep <= 0:
        return b""
    if len(data) <= keep:
        return data
    return data[-keep:]
