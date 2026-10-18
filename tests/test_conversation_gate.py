import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.llm.conversation import ConversationGate, closing_language
from app.llm.control import ControlMemory
import app.main as runtime
from test_shared_requests import server


@pytest.mark.parametrize('text,body,language', [
    ('Aria', '', 'fr'), ('Bonjour, Aria !', '', 'fr'), ('Hey Aria?', '', 'en'),
    ('ARIA, quelle heure est-il ?', 'quelle heure est-il ?', 'fr'),
    ('Hello Aria, what time is it?', 'what time is it?', 'en'),
])
def test_activation(text, body, language):
    gate = ConversationGate(clock=lambda: 0)
    turn = gate.begin('a', text, 0)
    assert (turn.kind, turn.text, turn.language) == ('activated', body, language)
    gate.finish(turn)
    assert gate.begin('a', 'What is two plus two?', 1).kind == 'followup'
    assert gate.begin('b', 'What is two plus two?', 1).kind == 'ignored'


@pytest.mark.parametrize('text', ['Maria, turn on the light', 'Ariana', 'Who is Aria?',
                                 'Please say Aria', "Aria's voice is nice", 'ariax'])
def test_mentions_and_similar_names_do_not_activate(text):
    gate = ConversationGate()
    assert gate.begin('a', text, 0).kind == 'ignored'
    assert not gate.sessions


@pytest.mark.parametrize('text,language', [
    ('Merci, au revoir Aria !', 'fr'), ('À bientôt.', 'fr'), ('Bonne nuit', 'fr'),
    ('Thanks, goodbye Aria.', 'en'), ("That's all.", 'en'), ('See you later', 'en'),
])
def test_explicit_goodbye_closes(text, language):
    gate = ConversationGate(clock=lambda: 0)
    gate.begin('a', 'Aria', 0)
    turn = gate.begin('a', text, 1)
    assert turn.kind == 'closed' and turn.language == language and turn.reset == ['a']
    assert gate.begin('a', 'Turn it off', 2).kind == 'ignored'


@pytest.mark.parametrize('text', ['Merci', 'Thanks', 'Explain the word goodbye',
                                 'Say au revoir in English', "Don't say goodbye"])
def test_goodbye_is_a_complete_control_utterance(text):
    assert closing_language(text) is None


def test_idle_boundary_uses_arrival_not_queue_processing_time():
    now = [0.]
    gate = ConversationGate(clock=lambda: now[0])
    gate.finish(gate.begin('a', 'Aria', 0))
    now[0] = 100  # Work waited behind another session.
    timely = gate.begin('a', 'Follow up', 29.9)
    assert timely.kind == 'followup'
    gate.finish(timely)
    expired = gate.begin('a', 'Follow up', 130)
    assert expired.kind == 'ignored' and expired.expired and expired.reset == ['a']
    assert gate.begin('a', 'Aria, question', 131).kind == 'activated'


@pytest.mark.asyncio
async def test_timeout_starts_after_playback_and_old_callbacks_cannot_reopen():
    now = [0.]
    gate = ConversationGate(clock=lambda: now[0])
    playback = asyncio.get_running_loop().create_future()
    turn = gate.begin('a', 'Aria', 0)
    gate.finish(turn, playback)
    now[0] = 60
    assert gate.sessions['a'].busy
    playback.set_result(None)
    await asyncio.sleep(0)
    assert gate.sessions['a'].ready_at == 60
    followup = gate.begin('a', 'question', 89)
    stale = asyncio.get_running_loop().create_future()
    gate.finish(followup, stale)
    gate.begin('a', 'Goodbye', 90)
    stale.set_result(None)
    await asyncio.sleep(0)
    assert 'a' not in gate.sessions


def test_no_shared_anonymous_session_and_bounded_state():
    gate = ConversationGate(limit=2, clock=lambda: 0)
    gate.finish(gate.begin(None, 'Aria', 0))
    assert gate.begin(None, 'Follow up', 1).kind == 'ignored'
    for key in ['a', 'b']:
        gate.finish(gate.begin(key, 'Aria', 0))
    third = gate.begin('c', 'Aria', 1)
    assert 'a' in third.reset and list(gate.sessions) == ['b', 'c']


@pytest.mark.parametrize('idle', [0, -1, float('nan'), float('inf'), 601])
def test_invalid_timeout_fails_closed(idle):
    with pytest.raises(ValueError):
        ConversationGate(idle_s=idle)


@pytest.mark.asyncio
async def test_shared_http_voice_gate_clears_context_and_preserves_raw_request(server, monkeypatch):
    client, state = server
    now = [time.monotonic()]
    state.conversation_gate = ConversationGate(clock=lambda: now[0])
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '8')
    inference = AsyncMock(return_value={'content': '4'})
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    # Queue timestamps and the deterministic clock must share the monotonic domain.
    async def request(text, source='api', session='same'):
        job = state.llm_queue.submit(text=text, session_id=session, speaker_user='known',
                                    speak_response=True, use_tools=True, source=source)
        job.queued_at = now[0]
        return await job.result
    sleeping = await request('Allume la lumière', 'voice')
    assert sleeping['ignored'] and not sleeping['spoken']
    inference.assert_not_awaited()
    wake = await request('Bonjour Aria')
    assert wake['response'] == 'Oui ?' and wake['tool_calls_made'] == 0
    first = await request('Combien font deux plus deux ?', 'voice')
    assert first['response'] == '4'
    assert len(state.llm_history['same']) == 2
    isolated = await request('What is two plus two?', session='other')
    assert isolated['ignored']
    state.control_memory = ControlMemory()
    state.control_memory.remember(('same', 'known'), 'old-device')
    closed = await request('Merci, au revoir', 'voice')
    assert closed['response'] == 'Au revoir.'
    assert 'same' not in state.llm_history and not state.control_memory.items
    assert (await request('Turn it off'))['ignored']
    reopened = await request('Aria, what is two plus two?')
    assert reopened['text'] == 'Aria, what is two plus two?'
    assert inference.call_args.kwargs['text'] == 'what is two plus two?'
    assert len(inference.call_args.kwargs['messages']) == 2
    now[0] += 30
    assert (await request('Follow up'))['ignored']
    assert 'same' not in state.llm_history


@pytest.mark.asyncio
async def test_failed_turn_does_not_leave_conversation_busy_forever(server, monkeypatch):
    client, state = server
    state.conversation_gate = ConversationGate()
    monkeypatch.setattr(runtime, 'generate_llm_response', AsyncMock(side_effect=RuntimeError('model error')))
    result = await client.post('/api/llm/process', json={'text': 'Aria, explain rain', 'session_id': 'failure'})
    assert result.status_code == 500
    assert not state.conversation_gate.sessions['failure'].busy


@pytest.mark.asyncio
async def test_http_session_identifier_is_bounded(server):
    client, _ = server
    result = await client.post('/api/llm/process', json={'text': 'Aria', 'session_id': 'x' * 257})
    assert result.status_code == 400


@pytest.mark.asyncio
async def test_voice_onset_survives_asr_delay_without_changing_queue_age(server, monkeypatch):
    _, state = server
    now = time.monotonic()
    state.conversation_gate = ConversationGate()
    # The last reply ended 35 seconds ago, but speech began at second 29.
    state.conversation_gate.finish(state.conversation_gate.begin('onset', 'Aria', now-35))
    state.conversation_gate.sessions['onset'].ready_at = now-35
    inference = AsyncMock(return_value={'content': '4'})
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    job = runtime._enqueue_final_llm(runtime.app, 'What is two plus two?', 'onset', 'known', started_at=now-6)
    assert job.conversation_started_at == now-6 and job.queued_at >= now
    assert (await job.result)['conversation_action'] == 'followup'
    inference.assert_awaited_once()
    state.conversation_gate.sessions['onset'].ready_at = now-35
    late = runtime._enqueue_final_llm(runtime.app, 'Turn it off', 'onset', 'known', started_at=now-4)
    assert (await late.result)['ignored']
    assert inference.await_count == 1


@pytest.mark.asyncio
async def test_websocket_forwards_server_observed_voice_onset(server, monkeypatch):
    import json
    _, state = server
    state.conversation_gate = ConversationGate()
    transcribed_at = []
    def transcribe(_):
        transcribed_at.append(time.monotonic())
        return 'Hello Aria'
    state.asr = SimpleNamespace(transcribe=transcribe)
    state.spk_recognizer = None
    state.echo_guard = None
    monkeypatch.setattr(runtime, '_should_run_asr', lambda _: (True, {'seconds': 1., 'rms': .1, 'speech_ratio': 1.}))
    original = runtime._enqueue_final_llm
    admitted = []
    def enqueue(*args, **kwargs):
        job = original(*args, **kwargs)
        admitted.append(job)
        return job
    monkeypatch.setattr(runtime, '_enqueue_final_llm', enqueue)
    messages = asyncio.Queue()
    for data in [
        {'type': 'start', 'sample_rate': 16000, 'channels': 1, 'format': 'pcm_s16le', 'chunk_ms': 40, 'client_vad': True},
        {'type': 'utterance_start'}, b'\0\0'*16000, {'type': 'utterance_end'},
    ]:
        messages.put_nowait({'type': 'websocket.receive', **({'bytes': data} if isinstance(data, bytes) else {'text': json.dumps(data)})})
    async def send_json(message):
        if message['status'] == 'completed':
            messages.put_nowait({'type': 'websocket.disconnect'})
    ws = SimpleNamespace(accept=AsyncMock(), close=AsyncMock(), receive=messages.get, send_json=send_json)
    before = time.monotonic()
    await asyncio.wait_for(runtime.ws_asr(ws), 2)
    job, = admitted
    assert before <= job.conversation_started_at <= transcribed_at[0] <= job.queued_at
    assert job.result.result()['conversation_action'] == 'activated'


def test_benchmark_transcript_equivalence_preserves_name_and_asr_errors():
    from scripts.e2e.wake_bench import normalized_transcript
    assert normalized_transcript('douze fois sept') == normalized_transcript('12 × 7')
    assert normalized_transcript('twelve times seven') == normalized_transcript('12 x 7')
    assert normalized_transcript('de plus 2') != normalized_transcript('deux plus deux')
    assert normalized_transcript('Hello area') != normalized_transcript('Hello Aria')
