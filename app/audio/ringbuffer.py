"""Bounded audio buffer.

This is used to ensure ARIA never buffers unbounded audio.
When over capacity, the oldest audio is dropped to keep latency bounded.
"""

from __future__ import annotations


class RingBuffer:
    def __init__(
        self,
        *,
        max_seconds: float,
        sample_rate: int,
        channels: int,
        sample_width_bytes: int = 2,
    ) -> None:
        if max_seconds <= 0:
            raise ValueError("max_seconds must be > 0")
        if sample_rate <= 0 or channels <= 0 or sample_width_bytes <= 0:
            raise ValueError("Invalid audio format")

        self._max_bytes = int(max_seconds * sample_rate * channels * sample_width_bytes)
        if self._max_bytes <= 0:
            raise ValueError("Computed max buffer size is <= 0")

        self._buf = bytearray()

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def append(self, data: bytes) -> int:
        """Append bytes. Returns number of bytes dropped (oldest) to stay bounded."""

        if not data:
            return 0

        dropped = 0
        self._buf.extend(data)
        overflow = len(self._buf) - self._max_bytes
        if overflow > 0:
            del self._buf[:overflow]
            dropped = overflow
        return dropped

    def get_bytes(self) -> bytes:
        return bytes(self._buf)
