from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from app.audio_sink.base import AudioSink, PlaybackHandle


@dataclass(frozen=True, slots=True)
class SonosHttpSink(AudioSink):
    speaker_ip: str

    async def play_url(self, url: str) -> PlaybackHandle:
        handle = PlaybackHandle(id=uuid.uuid4().hex)

        def _play() -> None:
            from soco import SoCo

            SoCo(self.speaker_ip).play_uri(url)

        await asyncio.to_thread(_play)
        return handle

    async def stop(self, handle: PlaybackHandle) -> None:
        # Sonos stop is global on the device; handle is currently informational.
        await self.stop_all()

    async def stop_all(self) -> None:
        def _stop() -> None:
            from soco import SoCo

            SoCo(self.speaker_ip).stop()

        await asyncio.to_thread(_stop)
