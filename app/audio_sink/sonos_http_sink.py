from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Iterable

from app.aria_logging import get_logger

from app.audio_sink.base import AudioSink, PlaybackHandle


log = get_logger("ARIA.Sonos")


class SonosDiscoveryError(RuntimeError):
    """Raised when automatic Sonos discovery fails."""


def _sorted_speakers(speakers: Iterable[object]) -> list[object]:
    return sorted(speakers, key=lambda s: ((getattr(s, "player_name", "") or "").lower(), getattr(s, "ip_address", "")))


def list_available_speakers(timeout: float = 5.0) -> list[tuple[str, str]]:
    try:
        from soco.discovery import discover
    except Exception as exc:  # pragma: no cover - SoCo is a declared dependency
        log.error("ARIA.SONOS.DiscoveryUnavailable", extra={"fields": {"error": repr(exc)}})
        return []

    try:
        speakers = discover(timeout=timeout) or set()
    except Exception as exc:
        log.warning(
            "ARIA.SONOS.DiscoveryFailed",
            extra={"fields": {"error": repr(exc), "timeout": timeout}},
        )
        return []

    entries: list[tuple[str, str]] = []
    for sp in _sorted_speakers(speakers):
        name = getattr(sp, "player_name", "") or "<unnamed>"
        ip = getattr(sp, "ip_address", "")
        entries.append((name, ip))
    return entries


def format_available_speakers(timeout: float = 5.0) -> str:
    entries = list_available_speakers(timeout=timeout)
    if not entries:
        return ""
    return ", ".join(f"{name} @ {ip}" for name, ip in entries)


def _availability_suffix(timeout: float) -> str:
    summary = format_available_speakers(timeout=timeout)
    if summary:
        return f" Available speakers: {summary}"
    return " No Sonos speakers discovered."


def verify_sonos_reachable(ip: str, timeout: float = 5.0) -> str:
    if not ip:
        raise SonosDiscoveryError("Sonos IP must not be empty.")

    try:
        from soco import SoCo
    except Exception as exc:  # pragma: no cover - dependency guaranteed by pyproject
        log.error("ARIA.SONOS.VerifyUnavailable", extra={"fields": {"error": repr(exc)}})
        raise SonosDiscoveryError("SoCo is not available; cannot verify Sonos speaker reachability.") from exc

    try:
        speaker = SoCo(ip)
        _ = speaker.uid  # Force a round-trip to ensure the device responds.
        name = speaker.player_name or ""
    except Exception as exc:
        log.warning(
            "ARIA.SONOS.VerifyFailed",
            extra={"fields": {"ip": ip, "error": repr(exc), "timeout": timeout}},
        )
        raise SonosDiscoveryError(f"Unable to reach Sonos speaker at {ip}.{_availability_suffix(timeout)}") from exc

    log.info("ARIA.SONOS.Verified", extra={"fields": {"ip": ip, "name": name or "<unknown>"}})
    return name


def discover_sonos_ip(*, preferred_name: str | None = None, timeout: float = 5.0) -> str:
    """Discover a Sonos speaker IP using SoCo discovery helpers."""

    try:
        from soco.discovery import by_name, discover
    except Exception as exc:  # pragma: no cover - SoCo is a declared dependency
        log.error("ARIA.SONOS.DiscoveryUnavailable", extra={"fields": {"error": repr(exc)}})
        raise SonosDiscoveryError("SoCo is not available; cannot auto-detect Sonos speakers.") from exc

    speaker = None
    if preferred_name:
        speaker = by_name(preferred_name, timeout=timeout)
        if speaker is None:
            log.warning(
                "ARIA.SONOS.NamedSpeakerMissing",
                extra={"fields": {"name": preferred_name, "timeout": timeout}},
            )
            raise SonosDiscoveryError(
                f"No Sonos speaker found with name '{preferred_name}'.{_availability_suffix(timeout)}"
            )

    if speaker is None:
        speakers = discover(timeout=timeout) or set()
        if not speakers:
            log.error(
                "ARIA.SONOS.NoSpeakers",
                extra={"fields": {"timeout": timeout}},
            )
            raise SonosDiscoveryError(f"No Sonos speakers discovered on the network.{_availability_suffix(timeout)}")
        speaker = sorted(speakers, key=lambda s: (s.player_name or "", s.ip_address))[0]

    mode = "by_name" if preferred_name else "auto"
    log.info(
        "ARIA.SONOS.Discovery",
        extra={"fields": {"ip": speaker.ip_address, "name": speaker.player_name, "mode": mode}},
    )
    return speaker.ip_address


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
