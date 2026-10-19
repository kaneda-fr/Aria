"""Sentence voice routing and bounded preparation; no real audio or inference."""
import asyncio
import threading
from types import SimpleNamespace

import pytest
from tests.test_speech_language import output
from app.speech_output.speech_output import speech_chunks
from app.tts.text_chunker import ChunkingConfig, chunk_text, _split_sentences


@pytest.mark.parametrize('text,hint,expected', [
    ('La porte est ouverte. The light is on.', 'fr', ['fr','en']),
    ('The door is open. La lampe est allumée.', 'en', ['en','fr']),
    ('La porte est ouverte. La lampe est allumée.', 'fr', ['fr']),
    ('The door is open. The light is on.', 'en', ['en']),
    ('Bonjour. The door is open.', 'fr', ['fr','en']),
    ('Hello. La porte est ouverte.', 'en', ['en','fr']),
    ('The door is open. 42.', 'fr', ['en']),
    ('La porte est ouverte. OK.', 'en', ['fr']),
    ('Le streaming audio est suspendu pour cinq minutes.', 'fr', ['fr']),
    ('The Wi-Fi router is offline.', 'en', ['en']),
    ('« La porte est ouverte. » The light is on.', 'fr', ['fr','en']),
    ('Dr. Smith says the light is on. La porte est ouverte.', 'fr', ['en','fr']),
])
def test_sentence_voice_decisions(text, hint, expected):
    chunks = speech_chunks(text, ChunkingConfig(enabled=False), hint)
    assert [lang for _,lang in chunks] == expected
    assert ' '.join(t for t,_ in chunks) == text


@pytest.mark.parametrize('hint,text', [
    ('fr', 'La porte est ouverte. La lampe est allumée. ' * 10),
    ('en', 'The door is open. The light is on. ' * 10),
])
@pytest.mark.parametrize('enabled', [False,True])
def test_monolingual_text_and_cache_boundaries_unchanged(hint,text,enabled):
    cfg=ChunkingConfig(enabled=enabled)
    assert speech_chunks(text,cfg,hint)==[(c,hint) for c in chunk_text(text,cfg)]


def test_decimal_abbreviation_and_closing_quote():
    assert _split_sentences('Dr. Smith measured 3.14 volts. "The light is on." La porte est ouverte.') == [
        'Dr. Smith measured 3.14 volts.', '"The light is on."', 'La porte est ouverte.']


@pytest.mark.asyncio
async def test_mixed_reply_uses_both_voices_and_distinct_caches(output):
    speech,sink,synthesized=output
    for _ in range(2):
        await speech.speak('La porte est ouverte. The light is on.',language='fr')
        await speech._task
        assert await speech.audio_ready
    assert synthesized == [('La porte est ouverte.','french'),('The light is on.','english')]
    urls=[x.args[0] for x in sink.play_url.await_args_list]
    assert urls[0]==urls[2] and urls[1]==urls[3] and urls[0]!=urls[1]


@pytest.mark.asyncio
async def test_one_lookahead_waits_for_playback_and_keeps_input_blocked(output):
    speech,sink,synthesized=output
    deliveries=[]
    def play(url):
        done=asyncio.get_running_loop().create_future()
        deliveries.append(done)
        return done
    sink.play_url.side_effect=play
    await speech.speak('La porte est ouverte. The light is on. La fenêtre est fermée.',language='fr')
    try:
        for _ in range(1000):
            if len(synthesized)==2 and deliveries: break
            await asyncio.to_thread(lambda:None)
        assert len(synthesized)==2 and len(deliveries)==1
        assert speech.input_blocked and not speech.audio_ready.done()
        deliveries[0].set_result(None)
        for _ in range(1000):
            if len(synthesized)==3 and len(deliveries)==2: break
            await asyncio.to_thread(lambda:None)
        assert len(synthesized)==3 and len(deliveries)==2
        assert speech.input_blocked and not speech.audio_ready.done()
        deliveries[1].set_result(None)
        for _ in range(1000):
            if len(deliveries)==3: break
            await asyncio.to_thread(lambda:None)
        assert speech.audio_ready.result() is True
        deliveries[2].set_result(None)
        await speech._task
    finally:
        await speech.interrupt()


@pytest.mark.asyncio
async def test_mute_discards_prepared_following_voice(output):
    speech,sink,synthesized=output
    delivered=asyncio.get_running_loop().create_future()
    sink.play_url.return_value=delivered
    await speech.speak('La porte est ouverte. The light is on.',language='fr')
    try:
        for _ in range(1000):
            if len(synthesized)==2: break
            await asyncio.to_thread(lambda:None)
        assert len(synthesized)==2
        speech.set_output_muted(True)
        speech.set_output_muted(False)
        delivered.set_result(None)
        await speech._task
        assert sink.play_url.await_count==1
        assert await speech.audio_ready is False
    finally:
        await speech.interrupt()


@pytest.mark.asyncio
async def test_cancel_during_lookahead_never_delivers_next_voice(output):
    speech,sink,synthesized=output
    started=threading.Event();release=threading.Event();finished=threading.Event()
    original=speech._engine.synthesize_to_wav
    def synthesize(text,**kwargs):
        if kwargs['voice']=='english':
            started.set()
            assert release.wait(2)
            try:return original(text,**kwargs)
            finally:finished.set()
        return original(text,**kwargs)
    speech._engine.synthesize_to_wav=synthesize
    delivery=asyncio.get_running_loop().create_future()
    sink.play_url.return_value=delivery
    await speech.speak('La porte est ouverte. The light is on.',language='fr')
    ready=speech.audio_ready
    try:
        assert await asyncio.to_thread(started.wait,2)
        await speech.interrupt()
        release.set()
        assert await asyncio.to_thread(finished.wait,2)
        assert sink.play_url.await_count==1
        assert await ready is False
        assert not speech._playing
    finally:
        release.set()
        await speech.interrupt()


@pytest.mark.asyncio
async def test_lookahead_failure_marks_output_incomplete(output):
    speech,sink,synthesized=output
    original=speech._engine.synthesize_to_wav
    def synthesize(text,**kwargs):
        if kwargs['voice']=='english':raise RuntimeError('voice unavailable')
        return original(text,**kwargs)
    speech._engine.synthesize_to_wav=synthesize
    delivery=asyncio.get_running_loop().create_future()
    delivery.set_result(None)
    sink.play_url.return_value=delivery
    await speech.speak('La porte est ouverte. The light is on.',language='fr')
    await speech._task
    assert await speech.audio_ready is False
    assert sink.play_url.await_count==1
