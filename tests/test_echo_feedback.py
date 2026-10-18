"""Echo admission uses effective identity and capture time, never a new model call."""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.main as runtime
from app.speech_output.echo_guard_v2 import EchoGuardV2, EchoV2Config
from app.speaker_recognition.recognizer import SpeakerResult
from test_shared_requests import server
from test_speech_language import output
from app.speech_output import speech_output as speech_module


def guard():
    return EchoGuardV2(EchoV2Config(True, 2., 5, 20, .65, .75, None))


@pytest.mark.parametrize('spoken,heard', [
    ('Il est 16:47.', 'Il est six or quarante sept.'),
    ('It is 16:47.', 'It is four forty seven.'),
    ('Il est 16:49.', 'Il est seize heure-neuf.'),
    ('It is 16:47.', 'It is 16:47.'),
    ('Aucun appareil ne correspond à cette recherche. Précise le nom et la pièce.',
     'Aucun appareil ne correspond à cette recherche. Précise le nom et la pied à ces faits.'),
    ('Available controls for the study lamp: turn on, turn off.',
     'Available controls for the study lamp turn on turn of something else.'),
])
def test_unknown_echo_opening_survives_asr_delay_and_tail_changes(spoken, heard):
    echo = guard()
    echo.record_tts_event(spoken, 100., 102., 'reply')
    suppressed, info = echo.should_suppress(heard, 120., 'unknown', 104.)
    assert suppressed and info['reason'] == 'unknown_speaker_prefix'
    assert info['tts_id'] == 'reply' and info['capture_age_s'] == 16.


@pytest.mark.parametrize('speaker,heard,onset', [
    ('human', 'Il est seize heures.', 104.),
    ('unknown', 'Il est possible de changer la lumière ?', 104.),
    ('unknown', 'Aria, il est seize heures ?', 104.),
    ('unknown', 'Il est seize heures.', 111.),
    ('unknown', 'Il est seize heures.', 99.),
    ('unknown', 'Il est seize heures.', 121.),
    ('unknown', 'Il est seize heures.', 10.),
])
def test_prefix_needs_unknown_identity_matching_opening_and_recent_capture(speaker, heard, onset):
    echo = guard()
    echo.record_tts_event('Il est 16:47.', 100., 102., 'reply')
    assert not echo.should_suppress(heard, 120., speaker, onset)[0]


def test_english_non_numeric_continuation_is_not_clock_echo():
    echo = guard()
    echo.record_tts_event('It is 16:47.', 100., 102., 'reply')
    assert not echo.should_suppress('It is possible to turn off the lamp?', 120., 'unknown', 104.)[0]


def test_old_previous_event_cannot_borrow_new_reply_window():
    echo = guard()
    echo.record_tts_event('Aucun appareil ne correspond à cette recherche.', 10., 12., 'old')
    echo.record_tts_event('The weather is unavailable.', 100., 102., 'new')
    assert not echo.should_suppress('Aucun appareil ne correspond à cette recherche.', 105., 'unknown', 101.)[0]


def test_whole_text_echo_still_works_when_recording_started_during_playback():
    echo = guard()
    spoken = 'Several colourful butterflies gather around fragrant flowers beside the garden wall.'
    echo.record_tts_event(spoken, 100., 106., 'reply')
    suppressed, info = echo.should_suppress(spoken, 120., 'human', 101.)
    assert suppressed and info['reason'] == 'text_similarity'


def test_disable_and_allowlist_remain_effective():
    for cfg in [EchoV2Config(False, 2., 5, 20, .65, .75, None),
                EchoV2Config(True, 2., 5, 20, .65, .75, r'^Il est')]:
        echo = EchoGuardV2(cfg)
        echo.record_tts_event('Il est 16:47.', 100., 102., 'reply')
        assert not echo.should_suppress('Il est seize heures.', 105., 'unknown', 104.)[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('person,expected_admissions', [('aria', 0), ('human', 1)])
async def test_voice_path_rejects_uncertain_self_prefix_before_queue_but_preserves_person(
        server, monkeypatch, person, expected_admissions):
    _, state = server
    state.asr = SimpleNamespace(transcribe=lambda _: 'Il est seize heures.')
    state.spk_recognizer = SimpleNamespace(enabled=True, self_min_score=.75,
        self_names=('aria_fr',), identify=lambda *a, **kw:
        SpeakerResult(person + '_fr', .25, False, person, .1))
    state.echo_guard = guard()
    now = time.time()
    state.echo_guard.record_tts_event('Il est 16:47.', now - 1., now + 1., 'reply')
    monkeypatch.setattr(runtime, '_should_run_asr', lambda _:
        (True, dict(seconds=1., rms=.1, speech_ratio=1.)))
    messages = [dict(type='start', sample_rate=16000, channels=1,
                     format='pcm_s16le', chunk_ms=40, client_vad=True),
                dict(type='utterance_start'), b'\0\0' * 16000, dict(type='utterance_end')]
    async def receive():
        if not messages:
            return {'type': 'websocket.disconnect'}
        data = messages.pop(0)
        return {'type': 'websocket.receive', **({'bytes': data} if isinstance(data, bytes)
                                               else {'text': json.dumps(data)})}
    jobs = []
    original = runtime._enqueue_final_llm
    def enqueue(*args, **kwargs):
        job = original(*args, **kwargs)
        jobs.append(job)
        return job
    monkeypatch.setattr(runtime, '_enqueue_final_llm', enqueue)
    ws = SimpleNamespace(accept=AsyncMock(), close=AsyncMock(), receive=receive, send_json=AsyncMock())
    await runtime.ws_asr(ws)
    await asyncio.wait_for(state.llm_queue.pending.join(), 2)
    assert len(jobs) == expected_admissions
    if jobs:
        assert jobs[0].speaker_user == 'human'
    else:
        state.speech_output.speak.assert_not_awaited()


@pytest.mark.asyncio
async def test_echo_window_uses_wav_duration_even_for_short_numeric_reply(output, monkeypatch):
    speech, sink, _ = output
    echo = SimpleNamespace(set_speaking=lambda _: None, record_tts_event=lambda *args: records.append(args))
    records = []
    speech.echo_guard = echo
    monkeypatch.setattr(speech_module, '_wav_duration_seconds', lambda _: 5.25)
    await speech._run_speak('Il est 16:47.', 1., 'fr')
    assert records[0][2] - records[0][1] == pytest.approx(5.25)
    sink.play_url.assert_awaited_once()
