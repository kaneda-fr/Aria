"""Acknowledged player lifecycle without microphones, speakers or inference."""
import asyncio
import pytest


@pytest.mark.asyncio
async def test_delivery_requires_finish_after_start():
    from app.speech_output.playback import PlaybackBroker
    broker = PlaybackBroker()
    player = broker.register()
    clip, finished = broker.begin(1.)
    with pytest.raises(ValueError): broker.ack(player, clip, 'finished')
    broker.ack(player, clip, 'started')
    assert not finished.done()
    broker.ack(player, clip, 'finished')
    await finished
    with pytest.raises(ValueError): broker.ack(player, clip, 'started')


@pytest.mark.asyncio
async def test_player_loss_fails_pending_delivery():
    from app.speech_output.playback import PlaybackBroker
    broker = PlaybackBroker()
    player = broker.register()
    clip, finished = broker.begin(1.)
    with pytest.raises(ValueError): broker.register()
    with pytest.raises(ValueError): broker.ack('other', clip, 'started')
    broker.unregister(player)
    with pytest.raises(RuntimeError): await finished
    assert broker.begin(1.) is None


@pytest.mark.asyncio
async def test_expired_player_cannot_ack_new_delivery():
    from app.speech_output.playback import PlaybackBroker
    now = [0.]
    broker = PlaybackBroker(clock=lambda: now[0])
    old = broker.register()
    now[0] = 16.
    assert broker.begin(1.) is None
    new = broker.register()
    clip, finished = broker.begin(1.)
    with pytest.raises(ValueError): broker.ack(old, clip, 'started')
    broker.ack(new, clip, 'started')
    broker.ack(new, clip, 'failed')
    with pytest.raises(RuntimeError): await finished


@pytest.mark.asyncio
async def test_reconnect_releases_expired_players_pending_clip():
    from app.speech_output.playback import PlaybackBroker
    now = [0.]
    broker = PlaybackBroker(clock=lambda: now[0])
    broker.register()
    _, old_finished = broker.begin(1.)
    now[0] = 16.
    broker.register()
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(old_finished, .1)
