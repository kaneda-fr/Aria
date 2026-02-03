from __future__ import annotations

from dataclasses import dataclass

import pytest
import soco
import soco.discovery as soco_discovery

from app.audio_sink.sonos_http_sink import (
    SonosDiscoveryError,
    discover_sonos_ip,
    verify_sonos_reachable,
)


@dataclass(frozen=True)
class _DummySpeaker:
    ip_address: str
    player_name: str


def test_discover_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_by_name(name: str, timeout: float) -> _DummySpeaker | None:
        assert name == "Living Room"
        assert timeout == 10
        return _DummySpeaker("192.0.2.10", "Living Room")

    def fake_discover(timeout: float) -> set[_DummySpeaker]:  # pragma: no cover
        raise AssertionError("discover() should not be called when name is provided")

    monkeypatch.setattr(soco_discovery, "by_name", fake_by_name)
    monkeypatch.setattr(soco_discovery, "discover", fake_discover, raising=False)

    ip = discover_sonos_ip(preferred_name="Living Room", timeout=10)
    assert ip == "192.0.2.10"


def test_discover_auto_falls_back_to_first_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(soco_discovery, "by_name", lambda *_args, **_kwargs: None)

    speakers = {
        _DummySpeaker("192.0.2.40", "Kitchen"),
        _DummySpeaker("192.0.2.30", "Bedroom"),
    }
    monkeypatch.setattr(soco_discovery, "discover", lambda timeout: speakers)

    ip = discover_sonos_ip(timeout=1)
    assert ip == "192.0.2.30"  # Bedroom sorts before Kitchen


def test_discover_raises_when_no_speakers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(soco_discovery, "by_name", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(soco_discovery, "discover", lambda timeout: set())

    with pytest.raises(SonosDiscoveryError):
        discover_sonos_ip(timeout=0.5)


def test_discover_by_name_missing_lists_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(soco_discovery, "by_name", lambda *_args, **_kwargs: None)

    speakers = {
        _DummySpeaker("192.0.2.40", "Kitchen"),
        _DummySpeaker("192.0.2.30", "Bedroom"),
    }
    monkeypatch.setattr(soco_discovery, "discover", lambda timeout: speakers)

    with pytest.raises(SonosDiscoveryError) as excinfo:
        discover_sonos_ip(preferred_name="Office", timeout=0.5)

    assert "Available speakers: Bedroom @ 192.0.2.30" in str(excinfo.value)


def test_verify_sonos_reachable_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SoCo:
        def __init__(self, ip: str) -> None:
            self.ip_address = ip
            self.player_name = "Kitchen"
            self.uid = "RINCON_123"

    monkeypatch.setattr(soco, "SoCo", _SoCo)

    name = verify_sonos_reachable("192.0.2.50", timeout=0.1)
    assert name == "Kitchen"


def test_verify_sonos_reachable_failure_lists_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class _SoCo:
        def __init__(self, ip: str) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(soco, "SoCo", _SoCo)

    speakers = {
        _DummySpeaker("192.0.2.10", "Living Room"),
        _DummySpeaker("192.0.2.11", "Office"),
    }
    monkeypatch.setattr(soco_discovery, "discover", lambda timeout: speakers)

    with pytest.raises(SonosDiscoveryError) as excinfo:
        verify_sonos_reachable("192.0.2.250", timeout=0.5)

    assert "Living Room @ 192.0.2.10" in str(excinfo.value)
