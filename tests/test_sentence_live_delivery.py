"""Live adapter delivery handshake, without audio devices or inference."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from app.speech_output.playback import PlaybackBroker
from tests.test_live_logging import live


@pytest.mark.asyncio
async def test_live_sink_waits_for_registered_player_ack(live, monkeypatch):
    monkeypatch.setattr(live, 'PLAYBACK', PlaybackBroker())
    monkeypatch.setattr(live, '_wav_duration_seconds', lambda _: 1.)
    monkeypatch.setattr(live.CaptureSink, 'play_url', AsyncMock())
    player = (await live.register_player())['player_id']
    with pytest.raises(HTTPException) as conflict:
        await live.register_player()
    assert conflict.value.status_code == 409
    finished = await live.LiveSink().play_url('http://localhost/tts/' + 'a' * 32 + '.wav')
    clip = next(iter(live.PLAYBACK.pending))
    assert isinstance(finished, asyncio.Future) and not finished.done()
    with pytest.raises(HTTPException):
        await live.player_status(player, clip, {'status': 'finished'})
    await live.player_status(player, clip, {'status': 'started'})
    await live.player_heartbeat(player)
    await live.player_status(player, clip, {'status': 'finished'})
    await finished
    assert not live.PLAYBACK.pending
    await live.unregister_player(player)
    assert await live.LiveSink().play_url('http://localhost/tts/' + 'b' * 32 + '.wav') is None


@pytest.mark.asyncio
async def test_live_playback_state_uses_speech_state_without_new_main_helpers(live, monkeypatch):
    monkeypatch.setattr(live, 'PLAYBACK', PlaybackBroker())
    monkeypatch.setattr(live.app.state, 'speech_output',
                        SimpleNamespace(input_revision=7, input_blocked=True), raising=False)
    state = await live.playback_state()
    assert state == {'input_blocked': True, 'input_revision': 7,
                     'pending': 0, 'player_connected': False}
