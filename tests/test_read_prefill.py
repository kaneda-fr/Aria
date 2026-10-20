"""Read replies remain deterministic while optional cache work stays disposable."""
import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

import app.main as runtime
from app.llm import history_prefill as prefill
from app.llm.ollama_client import OllamaConfig
from test_history_prefill import setup
from test_shared_requests import server
from test_inventory_read_requests import snapshot


def residency(state, names):
    response = httpx.Response(200, json={'models': [{'name': n} for n in names]},
                              request=httpx.Request('GET', 'http://local/api/ps'))
    state.llm_client = SimpleNamespace(get=AsyncMock(return_value=response))


@pytest.mark.asyncio
@pytest.mark.parametrize('names', [[], ['other:latest'], ['fixture:latest']])
async def test_read_prefill_requires_resident_model_not_previous_session(monkeypatch, names):
    state, args = setup()
    state.llm_recent_session = 'different-client'
    state.llm_recent_at = 0
    residency(state, names)
    inference = AsyncMock(return_value={'content': 'discard', 'tool_calls': ['discard']})
    monkeypatch.setattr(prefill, 'generate_llm_response', inference)
    prefill.schedule_prefill(state, **args, server_read=True)
    await asyncio.sleep(0)
    state.llm_client.get.assert_not_awaited()
    args['audio_ready'].set_result(True)
    await state.history_prefill_task
    assert inference.await_count == int(bool(names and names[0] == 'fixture:latest'))
    assert args['messages'][-1]['content'] == 'Il est 08:30.'
    assert not prefill.PREFILL.get()


@pytest.mark.asyncio
async def test_immediate_followup_cancels_before_model_or_residency(monkeypatch):
    state, args = setup()
    residency(state, ['fixture:latest'])
    inference = AsyncMock()
    monkeypatch.setattr(prefill, 'generate_llm_response', inference)
    args['audio_ready'].set_result(True)
    prefill.schedule_prefill(state, **args, server_read=True)
    await asyncio.sleep(.03)
    await prefill.cancel_prefill(state)
    state.llm_client.get.assert_not_awaited()
    inference.assert_not_awaited()
    assert args['audio_ready'].result()


@pytest.mark.asyncio
async def test_different_client_cancels_active_read_cache(monkeypatch):
    state, args = setup()
    residency(state, ['fixture:latest'])
    entered, closed = asyncio.Event(), asyncio.Event()
    async def inference(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
    monkeypatch.setattr(prefill, 'generate_llm_response', inference)
    args['audio_ready'].set_result(True)
    prefill.schedule_prefill(state, **args, server_read=True)
    await asyncio.wait_for(entered.wait(), 1)
    state.llm_queue.pending.put_nowait(SimpleNamespace(session_id='other-client'))
    await prefill.cancel_prefill(state)
    assert closed.is_set()
    assert state.llm_queue.pending.qsize() == 1


@pytest.mark.asyncio
async def test_read_cache_keeps_exact_bounded_history_and_never_dispatches(
        server, snapshot, monkeypatch):
    client, state = server
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    state.llm_config = OllamaConfig('ollama', 'http://local', 'fixture', num_predict=80)
    residency(state, ['fixture:latest'])
    old = [{'role': role, 'content': f'{role}-{i}'}
           for i in range(3) for role in ('user', 'assistant')]
    state.llm_history['read-cache'] = deque(old, maxlen=6)
    state.speech_output.audio_ready = asyncio.get_running_loop().create_future()
    state.speech_output.audio_ready.set_result(True)
    reader = AsyncMock(return_value={'readings': []})
    provider = SimpleNamespace(name='inventory', is_started=True,
                               indexer=SimpleNamespace(get_snapshot=lambda: snapshot),
                               read_resolved_state=reader)
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [provider],
                                            get_plugin=lambda _: None)
    foreground = AsyncMock(side_effect=AssertionError('read must remain deterministic'))
    background = AsyncMock(return_value={'content': 'Never speak this',
                                         'tool_calls': [{'function': {'name': 'execute_command'}}]})
    monkeypatch.setattr(runtime, 'generate_llm_response', foreground)
    monkeypatch.setattr(prefill, 'generate_llm_response', background)
    result = (await client.post('/api/llm/process', json={
        'text': 'Quelles commandes pour la lampe canapé dans le salon ?',
        'session_id': 'read-cache', 'speak_response': True})).json()
    assert result['success']
    await state.history_prefill_task
    history = list(state.llm_history['read-cache'])
    assert history[:4] == old[2:]
    assert history[-1]['content'] == result['response']
    assert background.await_count == 3
    assert background.await_args.kwargs['messages'][1:-1] == history
    assert all(c.kwargs['config'].num_predict == 1 for c in background.await_args_list)
    foreground.assert_not_awaited()
    reader.assert_not_awaited()
    state.speech_output.speak.assert_awaited_once()
