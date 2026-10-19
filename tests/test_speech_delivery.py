"""Speech delivery/mute invariants with fake sinks; no physical audio."""
import asyncio
from types import SimpleNamespace
import pytest
from tests.test_speech_language import output
import app.speech_output.speech_output as speech_module


@pytest.mark.asyncio
async def test_output_mute_skips_generation_and_delivery_then_expires(output, monkeypatch):
    speech, sink, synthesized = output
    now = [10.]
    monkeypatch.setattr(speech_module, 'time', SimpleNamespace(monotonic=lambda: now[0], time=lambda: 1.))
    assert speech.set_output_muted(True)['muted']
    await speech.speak('This must stay silent.', language='en')
    assert await speech.audio_ready is False
    assert not speech.input_blocked
    assert synthesized == []
    sink.play_url.assert_not_awaited()
    sink.stop_all.assert_not_awaited()
    now[0] += 100
    assert speech.set_output_muted(True)['mute_remaining_s'] == 200  # Retry does not extend it.
    now[0] += 201
    assert not speech.output_status()['muted']
    await speech.speak('Bonjour.', language='fr')
    await speech.playback_done
    assert synthesized == [('Bonjour.', 'french')]
    sink.play_url.assert_awaited_once()
    sink.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_output_mute_during_synthesis_discards_even_after_unmute(output, monkeypatch):
    speech, sink, _ = output
    assert callable(getattr(speech, 'set_output_muted', None))
    original = speech._engine.synthesize_to_wav
    def synthesize(*args, **kwargs):
        speech.set_output_muted(True)
        speech.set_output_muted(False)
        return original(*args, **kwargs)
    speech._engine.synthesize_to_wav = synthesize
    await speech.speak('Never replay this reply.', language='en')
    await speech.playback_done
    assert await speech.audio_ready is False
    sink.play_url.assert_not_awaited()
    assert not speech.input_blocked


@pytest.mark.asyncio
async def test_mute_unmute_before_scheduled_task_starts_still_discards_it(output):
    speech, sink, synthesized = output
    await speech.speak('Scheduled but not delivered.', language='en')
    speech.set_output_muted(True)
    speech.set_output_muted(False)
    await speech.playback_done
    assert await speech.audio_ready is False
    assert synthesized == []
    sink.play_url.assert_not_awaited()


@pytest.mark.asyncio
async def test_muted_reply_does_not_wait_for_active_player_or_replay_on_unmute(output):
    speech, sink, _ = output
    started = asyncio.Event()
    finished = asyncio.get_running_loop().create_future()
    async def play(_):
        started.set()
        return finished
    sink.play_url.side_effect = play
    await speech.speak('Already sent.', language='en')
    await asyncio.wait_for(started.wait(), 2)
    active = speech.playback_done
    try:
        speech.set_output_muted(True)
        assert await asyncio.wait_for(speech.speak('Discard this.', language='en'), .1) is False
        assert await speech.audio_ready is False
        speech.set_output_muted(False)
        assert speech.playback_done is active and not active.cancelled()
        finished.set_result(None)
        await active
        assert sink.play_url.await_count == 1
        sink.stop_all.assert_not_awaited()
    finally:
        if not finished.done(): finished.set_result(None)
        await active


@pytest.mark.asyncio
async def test_mute_preserves_current_delivery_but_skips_remaining_chunks(output, monkeypatch):
    speech, sink, synthesized = output
    monkeypatch.setattr(speech_module, 'chunk_text', lambda *_: ['First.', 'Second.'])
    started = asyncio.Event()
    finished = asyncio.get_running_loop().create_future()
    async def play(_):
        started.set()
        return finished
    sink.play_url.side_effect = play
    await speech.speak('First. Second.', language='en')
    await asyncio.wait_for(started.wait(), 2)
    current = speech.playback_done
    speech.set_output_muted(True)
    assert not current.cancelled() and not current.done()
    assert speech.input_blocked  # Keep protection for audio already sent to the speaker.
    finished.set_result(None)
    await current
    assert synthesized == [('First.', 'english')]
    assert sink.play_url.await_count == 1
    sink.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_next_reply_waits_for_player_finish(output):
    speech, sink, _ = output
    delivered = []
    first_started = asyncio.Event()
    finished = asyncio.get_running_loop().create_future()
    async def play(url):
        delivered.append(url)
        first_started.set()
        return finished if len(delivered) == 1 else None
    sink.play_url = play
    await speech.speak('The first reply is here.', language='en')
    await asyncio.wait_for(first_started.wait(), 1)
    first = speech.playback_done
    next_reply = asyncio.create_task(speech.speak('The second reply is here.', language='en'))
    # A loop checkpoint independent of the output fixture's mocked sleep.
    await asyncio.wait({next_reply}, timeout=.02)
    try:
        assert not first.done()
        assert not next_reply.done()
        assert len(delivered) == 1
        assert speech.input_blocked
        finished.set_result(None)
        await next_reply
        await speech.playback_done
        assert len(delivered) == 2
        assert not first.cancelled()
    finally:
        if not finished.done(): finished.set_result(None)
        await asyncio.gather(next_reply, return_exceptions=True)
        await speech.interrupt()
