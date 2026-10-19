"""Review regressions: cancellation and skipped speech, no physical audio."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from tests.test_speech_language import output
from app.speech_output.playback import PlaybackBroker


@pytest.mark.asyncio
async def test_broker_cancel_before_wait_starts_releases_slot_and_future():
    broker = PlaybackBroker()
    player = broker.register()
    clip, task = broker.begin(1.)
    future = broker.pending[clip]['future']
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not broker.pending
    assert future.done()
    broker.unregister(player)
    player = broker.register()
    clip, task = broker.begin(1.)
    broker.ack(player, clip, 'started')
    broker.ack(player, clip, 'finished')
    await task


@pytest.mark.asyncio
async def test_interrupt_invalidates_waiters_including_schedule_lock(output):
    speech, sink, synthesized = output
    started = asyncio.Event()
    finished = asyncio.get_running_loop().create_future()
    async def play(_):
        started.set()
        return finished
    sink.play_url.side_effect = play
    await speech.speak('The first reply is here.', language='en')
    await asyncio.wait_for(started.wait(), 1)
    second = asyncio.create_task(speech.speak('The second reply is here.', language='en'))
    third = asyncio.create_task(speech.speak('La troisième réponse est ici.', language='fr'))
    await asyncio.wait({second, third}, timeout=.01)
    try:
        await speech.interrupt()
        assert await asyncio.wait_for(second, 1) is False
        assert await asyncio.wait_for(third, 1) is False
        assert sink.play_url.await_count == 1
        assert len(synthesized) == 1
        sink.play_url.side_effect = None
        sink.play_url.return_value = None
        assert await speech.speak('A new reply after stopping.', language='en') is True
        await speech.playback_done
        assert sink.play_url.await_count == 2
    finally:
        second.cancel(); third.cancel()
        await asyncio.gather(second, third, return_exceptions=True)
        await speech.interrupt()


@pytest.mark.asyncio
@pytest.mark.parametrize('direct', [False, True])
async def test_muted_reply_not_reported_spoken_or_attached_to_prior_playback(direct):
    import app.main as runtime
    from app.llm.conversation import ConversationGate, conversation_turn
    prior = asyncio.get_running_loop().create_future()
    speech = SimpleNamespace(speak=AsyncMock(return_value=False), playback_done=prior)
    gate = ConversationGate(clock=lambda: 0.)
    app = SimpleNamespace(state=SimpleNamespace(speech_output=speech, conversation_gate=gate))
    async def process(app, job):
        return {'success': True, 'response': 'Four.', 'spoken': await runtime._speak_reply(app, 'Four.', 'en')}
    job = SimpleNamespace(text='Aria' if direct else 'Aria, what is two plus two?',
                          source='api', queued_at=0., session_id='muted', request_id='test', speak_response=True)
    try:
        result = await conversation_turn(process)(app, job)
        assert result['spoken'] is False
        assert not gate.sessions['muted'].busy
        assert not prior.done()
    finally:
        prior.cancel()


@pytest.mark.asyncio
async def test_request_arriving_during_stop_does_not_start_until_a_new_request(output):
    speech, sink, synthesized = output
    stopping = asyncio.Event()
    release = asyncio.Event()
    async def stop():
        stopping.set()
        await release.wait()
    sink.stop_all.side_effect = stop
    interruption = asyncio.create_task(speech.interrupt())
    await asyncio.wait_for(stopping.wait(), 1)
    request = asyncio.create_task(speech.speak('This arrived during stop.', language='en'))
    try:
        await asyncio.wait({request}, timeout=.01)
        assert not request.done()
        release.set()
        await interruption
        assert await request is False
        assert not synthesized
    finally:
        release.set()
        await interruption
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)
