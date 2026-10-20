"""Real HTTP/voice handlers with deterministic inference and fake device execution."""
import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
import app.main as runtime
import app.llm.tools as tooling
from app.llm.request_queue import RequestQueue


@pytest.mark.asyncio
async def test_numeric_tts_language_follows_each_request_across_transports_and_sessions(server, monkeypatch):
    client, state = server
    monkeypatch.setenv('ARIA_TTS_LANG_DETECT', 'heuristic')
    monkeypatch.setattr(runtime, 'generate_llm_response', AsyncMock(return_value={'content': '4'}))
    for source, session, text in [
        ('api', 'same', 'What is two plus two?'),
        ('voice', 'same', 'Combien font deux plus deux ? Réponds avec le nombre.'),
        ('api', 'other', 'What is two plus two?'),
        ('voice', 'other', 'Combien font deux plus deux ? Réponds avec le nombre.'),
    ]:
        if source == 'api':
            result = (await client.post('/api/llm/process', json={
                'text': text, 'session_id': session, 'speak_response': True})).json()
        else:
            result = await runtime._enqueue_final_llm(runtime.app, text, session, 'unknown').result
        assert result['spoken'] and result['response'] == '4'
    assert [c.kwargs['language'] for c in state.speech_output.speak.await_args_list] == ['en', 'fr', 'en', 'fr']


@pytest.mark.asyncio
async def test_shortcut_passes_current_input_language(server, monkeypatch):
    client, state = server
    monkeypatch.setenv('ARIA_TTS_LANG_DETECT', 'heuristic')
    state.router = SimpleNamespace(classify=lambda _: SimpleNamespace(mode='chat', detected_lang='fr', confidence=1.))
    result = await client.post('/api/llm/process', json={'text': 'Bonjour', 'speak_response': True})
    assert result.json()['spoken']
    state.speech_output.speak.assert_awaited_once_with('Bonjour! Comment puis-je vous aider?', language='fr')






@pytest_asyncio.fixture
async def server(monkeypatch):
    from app.llm.conversation import ConversationGate
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '0')
    state = runtime.app.state
    original = state._state.copy()
    state.conversation_gate = ConversationGate(enabled=False)
    state.llm_queue = RequestQueue(on_event=runtime._request_event)
    state.llm_config = SimpleNamespace()
    state.llm_client = object()
    state.llm_history = {}
    state.llm_history_lock = asyncio.Lock()
    state.router = None
    state.speech_output = SimpleNamespace(speak=AsyncMock())
    state.plugin_registry = SimpleNamespace(
        get_plugins_by_type=lambda _: [SimpleNamespace(name='inventory')],
        get_plugin=lambda _: object())
    worker = asyncio.create_task(runtime._llm_worker(runtime.app))
    transport = httpx.ASGITransport(app=runtime.app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url='http://aria') as client:
            yield client, state
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError): await worker
        state._state.clear()
        state._state.update(original)


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['Turn on the kitchen light.', 'Allume la lumière de la cuisine.'])
async def test_voice_and_http_execute_identical_tool_sequence_once(server, monkeypatch, text):
    from app.plugins.inventory.models import DeviceRecord, CommandRecord
    from app.plugins.inventory.indexer import InventoryIndexer
    from unittest.mock import Mock
    client, state = server
    room = 'kitchen' if text.startswith('Turn') else 'cuisine'
    index = InventoryIndexer()
    commands = [CommandRecord('fixture-'+action, 'device', 'action', 'other',
                'light_'+action, None, None, set(), {'args': {'payload_template':'other'}})
                for action in ['on','off']]
    index.add_device(DeviceRecord('fixture','device','Test Lamp',room,'light',True,True,commands,set(),0))
    inventory = SimpleNamespace(name='inventory',is_started=True,indexer=index)
    actions = SimpleNamespace(is_started=True,execute_command=Mock(return_value={
        'success':True,'dry_run':True,'executed':False,'message':'Fake misleading completion'}))
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [inventory],
                                            get_plugin=lambda _: actions)
    inference = AsyncMock(side_effect=AssertionError('Explicit grounded control needs no model selection'))
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    response = await client.post('/api/llm/process', json={'text': text, 'speak_response': True})
    voice = runtime._enqueue_final_llm(runtime.app, text, 'voice', 'unknown')
    result = await asyncio.wait_for(voice.result, 2)
    assert response.status_code == 200
    assert result['response'] == response.json()['response']
    assert result['tool_calls_made'] == response.json()['tool_calls_made'] == 1
    assert result['action_outcome'] == response.json()['action_outcome'] == 'recorded'
    assert actions.execute_command.call_count == 2
    assert all(c.kwargs['cmd_id']=='fixture-on' for c in actions.execute_command.call_args_list)
    assert actions.execute_command.call_args_list[0] == actions.execute_command.call_args_list[1]
    assert 'Fake misleading completion' not in result['response']
    inference.assert_not_awaited()
    assert state.speech_output.speak.await_count == 2
    assert result['request_id'] != response.json()['request_id']


@pytest.mark.asyncio
async def test_http_busy_does_not_evict_waiting_voice(server, monkeypatch):
    client, state = server
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []
    async def inference(**kwargs):
        seen.append(kwargs['text'])
        entered.set()
        await release.wait()
        return {'content': 'ok'}
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    first = asyncio.create_task(client.post('/api/llm/process', json={'text': 'first'}))
    await entered.wait()
    voice = runtime._enqueue_final_llm(runtime.app, 'second', 'voice', 'unknown')
    rejected = await client.post('/api/llm/process', json={'text': 'third'})
    assert rejected.status_code == 429 and rejected.json()['error'] == 'busy'
    assert rejected.headers['retry-after'] == '1'
    release.set()
    assert (await first).status_code == 200
    assert (await voice.result)['status'] == 'completed'
    assert seen == ['first', 'second']


@pytest.mark.asyncio
async def test_router_chat_factual_request_uses_model_and_greeting_speaks(server, monkeypatch):
    client, state = server
    state.router = SimpleNamespace(classify=lambda _: SimpleNamespace(mode='chat', detected_lang='en', confidence=.99))
    inference = AsyncMock(return_value={'content': 'Paris is the capital of France.'})
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    factual = await client.post('/api/llm/process', json={'text': 'What is the capital of France?', 'speak_response': True})
    greeting = runtime._enqueue_final_llm(runtime.app, 'Hello!', 'voice', 'unknown')
    greeting_result = await greeting.result
    assert factual.json()['response'] == 'Paris is the capital of France.'
    assert factual.json()['spoken'] and greeting_result['spoken']
    assert greeting_result['response'] == 'Hello! How can I help you?'
    assert inference.await_count == 1
    assert state.speech_output.speak.await_count == 2


@pytest.mark.asyncio
async def test_unavailable_capabilities_not_advertised_and_speech_optional(server, monkeypatch):
    client, state = server
    state.plugin_registry = None
    inference = AsyncMock(return_value={'content': 'hello'})
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    api = await client.post('/api/llm/process', json={'text': 'hello'})
    voice = runtime._enqueue_final_llm(runtime.app, 'hello', 'voice', 'unknown')
    await voice.result
    assert not api.json()['spoken']
    assert all(c.kwargs['tools'] is None for c in inference.await_args_list)
    assert state.speech_output.speak.await_count == 1


@pytest.mark.asyncio
async def test_history_is_session_scoped_across_transports(server, monkeypatch):
    client, state = server
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '8')
    inference = AsyncMock(return_value={'content': 'remembered'})
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    await client.post('/api/llm/process', json={'text': 'my colour is blue', 'session_id': 'same'})
    voice = runtime._enqueue_final_llm(runtime.app, 'what colour?', 'same', 'unknown')
    await voice.result
    assert len(inference.await_args_list[1].kwargs['messages']) == 4
    await client.post('/api/llm/process', json={'text': 'what colour?', 'session_id': 'other'})
    assert len(inference.await_args_list[2].kwargs['messages']) == 2


@pytest.mark.asyncio
async def test_voice_websocket_receives_admission_and_terminal_status(server, monkeypatch):
    import json
    client, state = server
    state.asr = SimpleNamespace(transcribe=lambda _: 'Bonjour')
    state.spk_recognizer = None
    state.echo_guard = None
    monkeypatch.setattr(runtime, '_should_run_asr', lambda _: (True, {'seconds': 1., 'rms': .1, 'speech_ratio': 1.}))
    monkeypatch.setattr(runtime, 'generate_llm_response', AsyncMock(return_value={'content': 'Bonjour!'}))
    messages = asyncio.Queue()
    for data in [
        {'type': 'start', 'sample_rate': 16000, 'channels': 1, 'format': 'pcm_s16le', 'chunk_ms': 40, 'client_vad': True},
        {'type': 'utterance_start'}, b'\0\0'*16000, {'type': 'utterance_end'},
    ]:
        messages.put_nowait({'type': 'websocket.receive', **({'bytes': data} if isinstance(data, bytes) else {'text': json.dumps(data)})})
    statuses = []
    async def send_json(message):
        statuses.append(message)
        if message['status'] == 'completed':
            messages.put_nowait({'type': 'websocket.disconnect'})
    ws = SimpleNamespace(accept=AsyncMock(), close=AsyncMock(), receive=messages.get, send_json=send_json)
    await asyncio.wait_for(runtime.ws_asr(ws), 2)
    assert [s['status'] for s in statuses] == ['accepted', 'completed']
    assert statuses[0]['request_id'] == statuses[1]['request_id']
    assert statuses[0]['session_id'] == statuses[1]['session_id']
    assert all('text' not in s and 'response' not in s for s in statuses)


@pytest.mark.asyncio
async def test_http_expiry_returns_504_without_model_execution(server, monkeypatch):
    client, state = server
    state.llm_queue.max_wait_s = 1e-12
    inference = AsyncMock()
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    response = await client.post('/api/llm/process', json={'text': 'stale command'})
    assert response.status_code == 504
    assert response.json()['status'] == 'expired'
    inference.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_http_waiter_never_executes_pending_action(server, monkeypatch):
    client, state = server
    entered, release = asyncio.Event(), asyncio.Event()
    calls, events = [], []
    state.llm_queue.on_event = events.append
    async def inference(**kwargs):
        calls.append(kwargs['text'])
        entered.set()
        await release.wait()
        return {'content': 'ok'}
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    voice = runtime._enqueue_final_llm(runtime.app, 'running', 'voice', 'unknown')
    await entered.wait()
    api = asyncio.create_task(client.post('/api/llm/process', json={'text': 'cancel me'}))
    await asyncio.sleep(0)
    api.cancel()
    with contextlib.suppress(asyncio.CancelledError): await api
    release.set()
    await voice.result
    await asyncio.wait_for(state.llm_queue.pending.join(), 1)
    assert calls == ['running']
    assert any(e['error'] == 'caller_cancelled' for e in events)


@pytest.mark.asyncio
async def test_model_failure_is_http_500_and_next_request_recovers(server, monkeypatch):
    client, state = server
    inference = AsyncMock(side_effect=[RuntimeError('model unavailable'), {'content': 'recovered'}])
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    failed = await client.post('/api/llm/process', json={'text': 'first'})
    recovered = await client.post('/api/llm/process', json={'text': 'second'})
    assert failed.status_code == 500 and failed.json()['status'] == 'failed'
    assert recovered.status_code == 200 and recovered.json()['response'] == 'recovered'


@pytest.mark.asyncio
async def test_model_action_on_conversation_is_blocked_and_reply_cannot_claim_success(server, monkeypatch):
    from unittest.mock import Mock
    client, state = server
    actions = SimpleNamespace(is_started=True, execute_command=Mock())
    state.plugin_registry = SimpleNamespace(
        get_plugins_by_type=lambda _: [SimpleNamespace(name='inventory',is_started=True)],
        get_plugin=lambda _: actions)
    inference = AsyncMock(return_value={'content':'The light is now on.', 'tool_calls':[
        {'function':{'name':'execute_command','arguments':{'cmd_id':'real-id'}}},
        {'function':{'name':'execute_command','arguments':{'cmd_id':'other-id'}}}]})
    monkeypatch.setattr(runtime,'generate_llm_response',inference)
    result = (await client.post('/api/llm/process',json={'text':'Explain rainbows.','speak_response':True})).json()
    actions.execute_command.assert_not_called()
    assert 'now on' not in result['response']
    assert result['tool_calls_made']==1 and result['spoken']
    assert inference.await_count==1


@pytest.mark.asyncio
async def test_local_clock_uses_shared_history_and_current_language_without_model(server, monkeypatch):
    client, state = server
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '8')
    monkeypatch.setenv('ARIA_TIMEZONE', 'UTC')
    inference = AsyncMock(side_effect=AssertionError('A clock answer must not call the model'))
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    api = (await client.post('/api/llm/process', json={
        'text': 'What time is it?', 'session_id': 'clock', 'speak_response': True, 'use_tools': False})).json()
    voice = await runtime._enqueue_final_llm(runtime.app, 'Quelle heure est-il ?', 'clock', 'unknown').result
    assert api['success'] and voice['success']
    assert api['tool_calls_made'] == voice['tool_calls_made'] == 0
    assert len(state.llm_history['clock']) == 4
    assert state.llm_history['clock'][-1]['content'] == voice['response']
    assert [call.kwargs['language'] for call in state.speech_output.speak.await_args_list] == ['en', 'fr']
    inference.assert_not_awaited()


@pytest.mark.asyncio
async def test_capability_question_never_executes_tools_and_keeps_history(server, monkeypatch):
    from unittest.mock import Mock
    client, state = server
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '8')
    inventory = SimpleNamespace(name='inventory', is_started=True)
    actions = SimpleNamespace(is_started=True, execution_mode='log_only', execute_command=Mock())
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [inventory], get_plugin=lambda _: actions)
    inference = AsyncMock(side_effect=AssertionError('Capability descriptions must not invoke the model'))
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    api = (await client.post('/api/llm/process', json={
        'text': 'What tools do you have?', 'session_id': 'caps', 'speak_response': True})).json()
    voice = await runtime._enqueue_final_llm(runtime.app, 'Que peux-tu faire ?', 'caps', 'unknown').result
    assert api['tool_calls_made'] == voice['tool_calls_made'] == 0
    assert 'find_devices and execute_command.' in api['response']
    assert 'seulement enregistrées' in voice['response']
    assert len(state.llm_history['caps']) == 4
    assert [c.kwargs['language'] for c in state.speech_output.speak.await_args_list] == ['en', 'fr']
    actions.execute_command.assert_not_called()
    inference.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_request_cancels_prefill_and_history_remains_exact(server, monkeypatch):
    from app.llm import history_prefill as prefill
    from app.llm.ollama_client import OllamaConfig
    client, state = server
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '4')
    state.llm_config = OllamaConfig('ollama', 'http://local', 'fixture')
    entered, closed = asyncio.Event(), asyncio.Event()
    snapshots = []
    async def prime(**kwargs):
        snapshots.append(kwargs['messages'])
        entered.set()
        try: await asyncio.Event().wait()
        finally: closed.set()
    async def infer(**kwargs):
        if entered.is_set(): assert closed.is_set()
        return {'content': '4'}
    monkeypatch.setattr(prefill, 'generate_llm_response', prime)
    monkeypatch.setattr(runtime, 'generate_llm_response', infer)
    state.speech_output.audio_ready = asyncio.get_running_loop().create_future()
    state.speech_output.audio_ready.set_result(True)
    async def request(text, session='history'):
        return (await client.post('/api/llm/process', json={
            'text': text, 'session_id': session, 'speak_response': True, 'use_tools': False})).json()
    await request('What is two plus two?')
    clock = await request('What time is it in Zurich?')
    await asyncio.wait_for(entered.wait(), 1)
    assert snapshots[0][1:-1] == list(state.llm_history['history'])
    assert snapshots[0][-2]['content'] == clock['response']
    assert len(state.llm_history['history']) == 4
    await request('What is two plus two?', session='another')
    assert closed.is_set() and state.history_prefill_task is None
    assert len(state.llm_history['history']) == 4
    assert len(state.llm_history['another']) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('text,language', [
    ('Find the reading lamp.', 'en'),
    ('Cherche la lampe de lecture.', 'fr'),
])
async def test_lookup_uses_one_model_round_and_shared_history(server, monkeypatch, text, language):
    client, state = server
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '8')
    inference = AsyncMock(return_value={'content': 'Misleading generated preamble', 'tool_calls': [
        {'function': {'name': 'find_devices', 'arguments': {'query': 'Reading lamp'}}}]})
    result = {'success': True, 'candidates': [
        {'device_id': 'lamp', 'device_name': 'Reading lamp', 'room': 'Study',
         'cmd_id': 'secret-42', 'capability': 'light_on'}], 'total_found': 1}
    tool = AsyncMock(return_value=result)
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    monkeypatch.setattr('app.llm.tools.execute_tool', tool)
    response = await client.post('/api/llm/process', json={
        'text': text, 'session_id': 'lookup-http', 'speak_response': True})
    voice = runtime._enqueue_final_llm(runtime.app, text, 'lookup-voice', 'unknown')
    spoken = await asyncio.wait_for(voice.result, 2)
    reply = response.json()['response']
    assert response.status_code == 200 and spoken['response'] == reply
    assert inference.await_count == tool.await_count == 2  # One per transport.
    assert all(c.kwargs['tool_name'] == 'find_devices' for c in tool.await_args_list)
    assert 'secret-42' not in reply and 'Misleading' not in reply
    assert ('État actuel inconnu' if language == 'fr' else 'Current state unknown') in reply
    assert state.llm_history['lookup-http'][-1]['content'] == reply
    assert state.llm_history['lookup-voice'][-1]['content'] == reply
    assert all(c.kwargs['language'] == language for c in state.speech_output.speak.await_args_list)

@pytest.mark.asyncio
@pytest.mark.parametrize('text,language', [
    ('Quelle est la température de Meteo North ?', 'fr'),
    ('What is the temperature from Meteo North?', 'en'),
])
async def test_state_read_uses_one_model_round_and_shared_history(server, monkeypatch, text, language):
    client, state = server
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    reading = {'success': True, 'status': 'read', 'readings': [{
        'cmd_id': '11', 'device_name': 'Meteo North', 'status': 'ok',
        'metric': 'temperature', 'value': 12.5, 'unit': '°C', 'observed_at': None}]}
    reader = SimpleNamespace(name='inventory', is_started=True,
                             read_state=AsyncMock(return_value=reading))
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [reader],
                                            get_plugin=lambda _: object())
    inference = AsyncMock(return_value={'content': 'Invented 99 degrees', 'tool_calls': [
        {'function': {'name': 'get_device_state',
                      'arguments': {'query': 'Meteo North', 'metric': 'temperature'}}}]})
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    response = await client.post('/api/llm/process', json={
        'text': text, 'session_id': 'read-http', 'speak_response': True})
    voice = runtime._enqueue_final_llm(runtime.app, text, 'read-voice', 'unknown')
    spoken = await asyncio.wait_for(voice.result, 2)
    reply = response.json()['response']
    assert response.status_code == 200 and spoken['response'] == reply
    assert inference.await_count == 2
    assert '12.5' in reply and '99' not in reply
    assert state.llm_history['read-http'][-1]['content'] == reply
    assert state.llm_history['read-voice'][-1]['content'] == reply
    assert all(c.kwargs['language'] == language for c in state.speech_output.speak.await_args_list)


@pytest.mark.asyncio
async def test_lookup_batch_does_not_hide_rejected_action(server, monkeypatch):
    client, state = server
    inference = AsyncMock(return_value={'content': '', 'tool_calls': [
        {'function': {'name': 'find_devices', 'arguments': {'query': 'lamp'}}},
        {'function': {'name': 'execute_command', 'arguments': {'cmd_id': 'invented'}}}]})
    tool = AsyncMock(side_effect=[{'success': True, 'candidates': []},
                                {'success': False, 'error': 'not authorized'}])
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    monkeypatch.setattr('app.llm.tools.execute_tool', tool)
    response = await client.post('/api/llm/process', json={'text': 'Please help with my lamp'})
    assert response.status_code == 200
    assert tool.await_count == 2 and inference.await_count == 1
    assert 'Available controls' not in response.json()['response']
    assert not tool.await_args_list[-1].kwargs.get('authorization')
