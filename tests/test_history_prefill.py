import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.llm import history_prefill as prefill
from app.llm.ollama_client import OllamaConfig


def setup():
    state = SimpleNamespace(
        llm_config=OllamaConfig('ollama', 'http://local', 'fixture', num_predict=80),
        llm_client=object(), llm_queue=SimpleNamespace(pending=asyncio.Queue()),
        llm_recent_session='session', llm_recent_at=time.monotonic())
    args = dict(session_id='session', request_id='request',
                audio_ready=asyncio.get_running_loop().create_future(),
                messages=[{'role': 'system', 'content': 'exact system'},
                          {'role': 'user', 'content': '[speaker=person] Quelle heure est-il ?'},
                          {'role': 'assistant', 'content': 'Il est 08:30.'}],
                tools=[{'type': 'function', 'function': {'name': 'execute_command'}}])
    return state, args


@pytest.mark.asyncio
async def test_exact_snapshot_after_audio_only_and_output_discarded(monkeypatch):
    state, args = setup()
    called = AsyncMock(return_value={'content': 'untrusted', 'tool_calls': ['never dispatch']})
    monkeypatch.setattr(prefill, 'generate_llm_response', called)
    prefill.schedule_prefill(state, **args)
    await asyncio.sleep(0)
    called.assert_not_awaited()
    args['messages'][-1]['content'] = 'later history'
    args['tools'].clear()
    args['audio_ready'].set_result(True)
    assert await state.history_prefill_task is None
    payload = called.await_args.kwargs
    assert payload['messages'][-2]['content'] == 'Il est 08:30.'
    assert payload['messages'][-1] == {'role': 'user', 'content': ''}
    assert payload['tools'][0]['function']['name'] == 'execute_command'
    assert payload['config'].num_predict == 1 and state.llm_config.num_predict == 80
    assert not prefill.PREFILL.get()


@pytest.mark.asyncio
@pytest.mark.parametrize('condition', ['cold', 'other_session', 'pending', 'disabled', 'no_audio', 'provider'])
async def test_skips_ineligible_work(monkeypatch, condition):
    state, args = setup()
    if condition == 'cold': state.llm_recent_at -= 61
    if condition == 'other_session': state.llm_recent_session = 'another'
    if condition == 'pending': state.llm_queue.pending.put_nowait(object())
    if condition == 'disabled': monkeypatch.setenv('ARIA_HISTORY_PREFILL', '0')
    if condition == 'no_audio': args['audio_ready'] = None
    if condition == 'provider': state.llm_config = object()
    prefill.schedule_prefill(state, **args)
    assert getattr(state, 'history_prefill_task', None) is None




@pytest.mark.asyncio
async def test_cancellation_closes_inference_and_does_not_cancel_audio(monkeypatch):
    state, args = setup()
    entered, closed = asyncio.Event(), asyncio.Event()
    async def inference(**kwargs):
        assert prefill.PREFILL.get()
        entered.set()
        try: await asyncio.Event().wait()
        finally: closed.set()
    monkeypatch.setattr(prefill, 'generate_llm_response', inference)
    prefill.schedule_prefill(state, **args)
    args['audio_ready'].set_result(True)
    await asyncio.wait_for(entered.wait(), 1)
    task = state.history_prefill_task
    prefill.schedule_prefill(state, **args)
    assert state.history_prefill_task is task
    await prefill.cancel_prefill(state)
    assert closed.is_set() and task.cancelled() and state.history_prefill_task is None
    assert args['audio_ready'].result() is True


@pytest.mark.asyncio
async def test_cancel_while_waiting_does_not_poison_speech_future():
    state, args = setup()
    prefill.schedule_prefill(state, **args)
    await asyncio.sleep(0)
    await prefill.cancel_prefill(state)
    args['audio_ready'].set_result(True)
    assert args['audio_ready'].result()


@pytest.mark.asyncio
@pytest.mark.parametrize('condition', ['failed_audio', 'queued_after_schedule', 'model_failure', 'timeout'])
async def test_late_failures_do_not_escape_or_dispatch(monkeypatch, condition):
    state, args = setup()
    inference = AsyncMock(side_effect=TimeoutError() if condition == 'timeout' else RuntimeError())
    monkeypatch.setattr(prefill, 'generate_llm_response', inference)
    prefill.schedule_prefill(state, **args)
    if condition == 'queued_after_schedule': state.llm_queue.pending.put_nowait(object())
    args['audio_ready'].set_result(condition != 'failed_audio')
    await state.history_prefill_task
    assert inference.await_count == (1 if condition in {'model_failure', 'timeout'} else 0)
