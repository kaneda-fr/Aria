from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PlaybackHandle:
    id: str


class AudioSink(Protocol):
    async def play_url(self, url: str) -> PlaybackHandle:
        ...

    async def stop(self, handle: PlaybackHandle) -> None:
        ...

    async def stop_all(self) -> None:
        ...
