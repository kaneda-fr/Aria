"""Test live instrumentation without leaking its app-global hooks to other tests."""
import asyncio
import importlib
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv('ARIA_E2E_RUN', str(tmp_path))
    import app.main as runtime
    import app.llm.tools as tooling
    import app.speech_output.speech_output as speech
    # The standalone adapter installs hooks at import time. These tests invoke
    # adapter functions directly; restore the shared production boundaries before
    # returning so the rest of the suite cannot inherit its wrappers or routes.
    originals = [(obj, name, getattr(obj, name)) for obj, name in (
        (runtime, '_process_request'), (tooling, 'execute_tool'),
        (speech, 'SonosHttpSink'), (speech, 'verify_sonos_reachable'),
        (runtime.app.router, 'lifespan_context'))]
    routes = list(runtime.app.router.routes)
    logger = logging.getLogger('ARIA')
    handlers = list(logger.handlers)
    try:
        module = importlib.import_module('scripts.e2e.live_server')
    finally:
        for obj, name, value in originals:
            setattr(obj, name, value)
        runtime.app.router.routes[:] = routes
        logger.handlers[:] = handlers
    monkeypatch.setattr(module, 'RUN', tmp_path)
    return module


def app_state():
    return SimpleNamespace(state=SimpleNamespace(
        spk_recognizer=SimpleNamespace(enabled=True, _ready=True),
        llm_client=SimpleNamespace(event_hooks={'response': []})))


def job():
    return SimpleNamespace(request_id='req-1', session_id='session-1',
                           source='voice', text='Écoute « ceci »\n' + 'long ' * 200,
                           speaker_user='fixture-user')


def entries(live):
    return [json.loads(line) for line in (live.RUN / 'conversation.jsonl').read_text().splitlines()]


@pytest.mark.asyncio
async def test_exact_requests_replies_and_context_survive_logging(live, monkeypatch):
    result = {'response': 'Voici ma réponse exacte.\n' + 'texte ' * 200,
              'success': True, 'error': None, 'spoken': True}
    pending = []
    monkeypatch.setattr(live.CaptureSink, 'play_url', AsyncMock())
    async def process(app, request):
        pending.append(asyncio.create_task(live.LiveSink().play_url('http://local/tts/fixture.wav')))
        return result
    monkeypatch.setattr(live, 'original_process', process)
    request = job()
    assert await live.logged_process(app_state(), request) is result
    assert live.REQUEST.get() is None
    await pending[0]
    events = entries(live)
    assert events[0]['text'] == request.text
    assert events[0]['speaker_status'] == 'attributed'
    assert events[1]['text'] == result['response']
    assert events[2]['event'] == 'Live.Audio'
    assert all(e['request_id'] == request.request_id for e in events)


@pytest.mark.asyncio
async def test_failure_recorded_without_success_or_leaked_request_context(live, monkeypatch):
    monkeypatch.setattr(live, 'original_process', AsyncMock(side_effect=RuntimeError('fixture failure')))
    state = app_state()
    state.state.spk_recognizer.enabled = False
    with pytest.raises(RuntimeError):
        await live.logged_process(state, job())
    assert [e['event'] for e in entries(live)] == ['Live.Request', 'Live.Error']
    assert entries(live)[0]['speaker_status'] == 'disabled'
    assert live.REQUEST.get() is None


@pytest.mark.asyncio
async def test_model_history_and_truncation_metadata(live):
    request = httpx.Request('POST', 'http://local/api/chat', json={
        'messages': [{'role': r, 'content': 'fixture'} for r in ['system','user','assistant','user']]})
    response = httpx.Response(200, request=request, json={
        'done_reason': 'length', 'eval_count': 80, 'prompt_eval_count': 100})
    token = live.REQUEST.set('req-2')
    try:
        await live.observe_model(response)
    finally:
        live.REQUEST.reset(token)
    entry = entries(live)[0]
    assert entry['request_id'] == 'req-2'
    assert entry['history_messages'] == 2
    assert entry['done_reason'] == 'length'


@pytest.mark.asyncio
async def test_readiness_speaks_without_model_or_history(live, monkeypatch):
    state = live.app.state
    monkeypatch.setattr(state, 'warmup_ready', True, raising=False)
    speak = AsyncMock(return_value=True)
    monkeypatch.setattr(live.runtime, '_speak_reply', speak)
    monkeypatch.setattr(live, 'original_process', AsyncMock(side_effect=AssertionError('No LLM on readiness')))
    assert (await live.ready_speech())['spoken']
    speak.assert_awaited_once_with(live.app, 'Aria is ready.', 'en')
    assert live.REQUEST.get() is None


@pytest.mark.asyncio
async def test_lifespan_does_not_yield_before_warmup(live, monkeypatch):
    from contextlib import asynccontextmanager
    marker = []
    @asynccontextmanager
    async def original(app):
        yield
    state = app_state()
    state.state.llm_config = object()
    state.state.llm_history = {}
    state.state.plugin_registry = SimpleNamespace(get_plugin=lambda name: None)
    monkeypatch.setattr(live, 'original_lifespan', original)
    monkeypatch.setattr(live, 'install_live_tools', lambda *args: None)
    async def prime(*args, **kwargs):
        assert not state.state.warmup_ready
        marker.append('primed')
        return {'tools': True}
    monkeypatch.setattr(live, 'warmup', prime)
    async with live.live_lifespan(state):
        assert marker == ['primed'] and state.state.warmup_ready
        assert state.state.llm_history == {}
    assert live.REQUEST.get() is None


@pytest.mark.asyncio
async def test_discarded_prefill_not_counted_as_reply_generation(live):
    request = httpx.Request('POST', 'http://local/api/chat', json={
        'messages': [{'role': 'system', 'content': 'fixture'}, {'role': 'user', 'content': ''}]})
    response = httpx.Response(200, request=request, json={'done_reason': 'length', 'eval_count': 1})
    token = live.PREFILL.set(True)
    try:
        await live.observe_model(response)
    finally:
        live.PREFILL.reset(token)
    assert entries(live)[0]['event'] == 'Live.Prefill'
