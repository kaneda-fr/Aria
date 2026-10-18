"""Request-local voice selection with fake Piper/Sonos; no device/network IO."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import app.speech_output.speech_output as speech_module
from app.tts.tts_engine import TtsConfig


@pytest.fixture
def output(monkeypatch, tmp_path):
    sink = SimpleNamespace(set_volume=AsyncMock(), play_url=AsyncMock(), stop_all=AsyncMock())
    synthesized = []
    def synthesize(text, *, voice, rate):
        synthesized.append((text, voice))
        return tmp_path / 'raw.wav'
    def normalize(raw, path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fake wav')
    monkeypatch.setenv('ARIA_SONOS_IP', '127.0.0.1')
    monkeypatch.setenv('ARIA_HTTP_BASE_URL', 'http://aria')
    monkeypatch.setenv('ARIA_TTS_VOICE_MAP', 'en:english,fr:french')
    monkeypatch.setenv('ARIA_TTS_LANG_DETECT', 'heuristic')
    monkeypatch.setenv('ARIA_TTS_CHUNKING', '0')
    monkeypatch.setattr(speech_module, 'build_tts_engine', lambda _: SimpleNamespace(synthesize_to_wav=synthesize))
    monkeypatch.setattr(speech_module, 'verify_sonos_reachable', lambda *a, **k: 'fake')
    monkeypatch.setattr(speech_module, 'SonosHttpSink', lambda **_: sink)
    monkeypatch.setattr(speech_module, 'normalize_for_sonos', normalize)
    monkeypatch.setattr(speech_module, '_wav_duration_seconds', lambda _: 0)
    monkeypatch.setattr(speech_module.asyncio, 'sleep', AsyncMock())
    speech = speech_module.SpeechOutput(TtsConfig(True, 'piper', 'english', 1., tmp_path, 2.))
    return speech, sink, synthesized


@pytest.mark.asyncio
async def test_alternating_numeric_replies_use_request_voice_and_separate_cache(output):
    speech, sink, synthesized = output
    for language in ('en', 'fr', 'en'):
        await speech.speak('4', language=language)
        await speech._task
    assert synthesized == [('4', 'english'), ('4', 'french')]
    urls = [c.args[0] for c in sink.play_url.await_args_list]
    assert urls[0] == urls[2] and urls[0] != urls[1]


@pytest.mark.asyncio
async def test_request_language_is_fixed_for_all_chunks(output, monkeypatch):
    speech, _, synthesized = output
    monkeypatch.setattr(speech_module, 'chunk_text', lambda *_: ['Bonjour.', '4', 'The end.'])
    await speech.speak('Bonjour. 4 The end.', language='fr')
    await speech._task
    assert synthesized == [('Bonjour.', 'french'), ('4', 'french'), ('The end.', 'french')]


@pytest.mark.asyncio
async def test_legacy_call_without_hint_does_not_inherit_previous_request(output):
    speech, _, synthesized = output
    await speech.speak('Bonjour', language='fr')
    await speech._task
    await speech.speak('4')
    await speech._task
    assert synthesized == [('Bonjour', 'french'), ('4', 'english')]


@pytest.mark.asyncio
async def test_invalid_hint_does_not_cancel_active_speech(output):
    speech, sink, _ = output
    prior = asyncio.create_task(asyncio.Event().wait())
    speech._task = prior
    try:
        with pytest.raises(ValueError):
            await speech.speak('4', language='invalid')
        assert speech._task is prior and not prior.cancelled()
        sink.stop_all.assert_not_awaited()
    finally:
        await speech.interrupt()


@pytest.mark.parametrize('text,expected', [
    ('Combien font deux plus deux ? Réponds uniquement avec le nombre.', 'fr'),
    ('What is two plus two? Answer with just the number.', 'en'),
    ('Quelle est la capitale de la France?', 'fr'),
    ('What is the capital of France?', 'en'),
    ('4', 'en'),
])
def test_input_language_detection_is_local_and_request_scoped(monkeypatch, text, expected):
    monkeypatch.setenv('ARIA_TTS_LANG_DETECT', 'heuristic')
    assert speech_module.detect_language(text) == expected


@pytest.mark.asyncio
async def test_ready_signal_waits_for_last_audio_delivery(output, monkeypatch):
    speech, sink, _ = output
    monkeypatch.setattr(speech_module, 'chunk_text', lambda *_: ['First.', 'Last.'])
    observed = []
    async def play(url):
        observed.append(speech.audio_ready.done())
    sink.play_url.side_effect = play
    await speech.speak('First. Last.')
    await speech._task
    assert observed == [False, False]
    assert speech.audio_ready.result() is True


@pytest.mark.asyncio
async def test_failed_speech_and_cancel_before_start_signal_false(output):
    speech, sink, _ = output
    sink.play_url.side_effect = RuntimeError('failed delivery')
    await speech.speak('Failure.')
    await speech._task
    await asyncio.sleep(0)
    assert await speech.audio_ready is False
    await speech.speak('Cancelled.')
    ready = speech.audio_ready
    await speech.interrupt()
    assert await ready is False
