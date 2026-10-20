"""History retention remains exact while speculative cache work prepares eviction."""
import asyncio
from collections import deque
from unittest.mock import AsyncMock

import pytest

from test_shared_requests import server
import app.main as runtime
from app.llm import history_prefill as prefill
from app.llm.ollama_client import OllamaConfig


@pytest.mark.asyncio
@pytest.mark.parametrize('prior_pairs', [1, 2, 3])
async def test_rollover_prepares_only_the_exact_retained_history(server, monkeypatch, prior_pairs):
    client, state = server
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    state.llm_config = OllamaConfig('ollama', 'http://local', 'fixture', num_predict=80)
    history = [{'role': role, 'content': f'{role}-{i}'}
               for i in range(prior_pairs) for role in ('user', 'assistant')]
    state.llm_history['session'] = deque(history, maxlen=6)
    state.speech_output.audio_ready = asyncio.get_running_loop().create_future()
    state.speech_output.audio_ready.set_result(True)
    inference = AsyncMock(return_value={'content': 'A new answer.'})
    speculative = AsyncMock(return_value={'content': 'Never speak or store this.',
                                        'tool_calls': [{'function': {'name': 'execute_command'}}]})
    monkeypatch.setattr(runtime, 'generate_llm_response', inference)
    monkeypatch.setattr(prefill, 'generate_llm_response', speculative)
    result = await client.post('/api/llm/process', json={
        'session_id': 'session', 'text': 'Explain a rainbow.', 'speak_response': True})
    assert result.json()['response'] == 'A new answer.'
    expected = (history + [{'role': 'user', 'content': '[speaker=unknown] Explain a rainbow.'},
                           {'role': 'assistant', 'content': 'A new answer.'}])[-6:]
    assert list(state.llm_history['session']) == expected
    if prior_pairs == 3:
        task = getattr(state, 'history_prefill_task', None)
        assert task is not None, 'Eviction must prepare the changed prefix after audio readiness'
        await task
        assert speculative.await_args.kwargs['messages'] == [
            {'role': 'system', 'content': runtime.SYSTEM_PROMPT}, *expected,
            {'role': 'user', 'content': ''}]
        assert list(state.llm_history['session']) == expected
        state.speech_output.speak.assert_awaited_once()
    else:
        assert getattr(state, 'history_prefill_task', None) is None
        speculative.assert_not_awaited()


@pytest.mark.asyncio
async def test_rollover_prefill_can_finish_beyond_local_answer_budget(monkeypatch):
    from test_history_prefill import setup
    state, args = setup()
    completed = asyncio.Event()

    async def inference(**kwargs):
        await asyncio.sleep(4.05)
        completed.set()
        return {'content': 'discarded'}

    monkeypatch.setattr(prefill, 'generate_llm_response', inference)
    prefill.schedule_prefill(state, **args, rollover=True)
    args['audio_ready'].set_result(True)
    await state.history_prefill_task
    assert completed.is_set(), 'Rollover prefill must survive the old four-second timeout'


@pytest.mark.asyncio
async def test_next_request_cancels_rollover_work(server, monkeypatch):
    client, state = server
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '2')
    state.llm_config = OllamaConfig('ollama', 'http://local', 'fixture')
    state.llm_history['session'] = deque([
        {'role': 'user', 'content': 'previous'}, {'role': 'assistant', 'content': 'answer'}], maxlen=2)
    state.speech_output.audio_ready = asyncio.get_running_loop().create_future()
    state.speech_output.audio_ready.set_result(True)
    entered, closed = asyncio.Event(), asyncio.Event()

    async def speculative(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(prefill, 'generate_llm_response', speculative)
    monkeypatch.setattr(runtime, 'generate_llm_response', AsyncMock(return_value={'content': '4'}))
    body = {'session_id': 'session', 'text': 'What is two plus two?', 'speak_response': True}
    await client.post('/api/llm/process', json=body)
    await asyncio.wait_for(entered.wait(), 1)
    task = state.history_prefill_task
    body['speak_response'] = False
    response = await asyncio.wait_for(client.post('/api/llm/process', json=body), 1)
    assert response.json()['response'] == '4'
    assert closed.is_set() and task.cancelled()
    assert state.history_prefill_task is None


@pytest.mark.asyncio
async def test_rollover_progresses_through_short_exact_prefixes(monkeypatch):
    from test_history_prefill import setup
    state, args = setup()
    history = [{'role': role, 'content': f'{role}-{i}'}
               for i in range(6) for role in ('user', 'assistant')]
    args['messages'] = [{'role': 'system', 'content': 'exact system'}, *history]
    observed = []

    async def inference(**kwargs):
        observed.append(kwargs['messages'])
        return {'content': 'discarded', 'tool_calls': []}

    monkeypatch.setattr(prefill, 'generate_llm_response', inference)
    prefill.schedule_prefill(state, **args, rollover=True)
    args['audio_ready'].set_result(True)
    await state.history_prefill_task
    assert len(observed) == 6, 'A rollover must not submit the entire uncached history in one call'
    for i, snapshot in enumerate(observed, 1):
        assert snapshot == [args['messages'][0], *history[:2*i], {'role': 'user', 'content': ''}]
    assert args['messages'] == [{'role': 'system', 'content': 'exact system'}, *history]


@pytest.mark.asyncio
async def test_pending_request_prevents_next_rollover_step(monkeypatch):
    from test_history_prefill import setup
    state, args = setup()
    args['messages'] += [{'role': 'user', 'content': 'second'},
                         {'role': 'assistant', 'content': 'answer'}]
    observed = []

    async def inference(**kwargs):
        observed.append(kwargs['messages'])
        state.llm_queue.pending.put_nowait(object())
        return {'content': 'discarded'}

    monkeypatch.setattr(prefill, 'generate_llm_response', inference)
    prefill.schedule_prefill(state, **args, rollover=True)
    args['audio_ready'].set_result(True)
    await state.history_prefill_task
    assert len(observed) == 1
    assert observed[0][-2]['content'] == 'Il est 08:30.'
