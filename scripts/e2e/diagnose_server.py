"""Deterministic current-handler probes, with fake inference and no device IO.

Run in a separate process with Apollo dependencies; never import into a serving app.
The original defect observations are retained in baseline-2026-10-17.md.
"""
from __future__ import annotations
import asyncio
import contextlib
import importlib
import json
import os
from types import SimpleNamespace
from app.llm.request_queue import RequestQueue


async def diagnose():
    os.environ['ARIA_LLM_HISTORY_MAX_MESSAGES'] = '0'
    runtime = importlib.import_module('app.main')
    app = runtime.app
    app.state.llm_queue = RequestQueue()
    first = runtime._enqueue_final_llm(app, 'first request', 'session-A', 'unknown')
    second = runtime._enqueue_final_llm(app, 'second request', 'session-B', 'unknown')
    result = {'queue': {'first_status': first.status, 'second_result': second.result.result()}}
    assert first.status == 'queued' and second.result.result()['error'] == 'busy'
    app.state.llm_queue.finish(first, 'cancelled', error='probe_complete')
    app.state.llm_queue = RequestQueue()
    calls, speech = [], []
    async def inference(**kwargs):
        calls.append({'text': kwargs['text'], 'tools_offered': len(kwargs.get('tools') or [])})
        return {'content': 'Diagnostic reply.', 'tool_calls': None}
    async def speak(text, *, language=None): speech.append(text)
    runtime.generate_llm_response = inference
    app.state.llm_config = SimpleNamespace(url='fake', model='fake')
    app.state.llm_client = object()
    app.state.plugin_registry = SimpleNamespace(
        get_plugins_by_type=lambda _: [SimpleNamespace(name='inventory')],
        get_plugin=lambda _: object())
    app.state.speech_output = SimpleNamespace(speak=speak)
    app.state.router = None
    worker = asyncio.create_task(runtime._llm_worker(app))
    try:
        text = 'Turn on the light in the living room.'
        await runtime.process_text({'text': text, 'use_tools': True})
        job = runtime._enqueue_final_llm(app, text, 'voice-session', 'unknown')
        await asyncio.wait_for(asyncio.shield(job.result), 2)
        result['entrypoint_parity'] = {'api': calls[0], 'voice': calls[1]}
        app.state.router = SimpleNamespace(classify=lambda _: SimpleNamespace(
            mode='chat', detected_lang='en', confidence=.99))
        result['router_chat_branch'] = await runtime.process_text({
            'text': 'What is the capital of France?', 'speak_response': True})
        assert calls[0]['tools_offered'] == calls[1]['tools_offered'] == 2
        assert result['router_chat_branch']['response'] == 'Diagnostic reply.'
        assert result['router_chat_branch']['spoken'] is True
        result['scope'] = 'Handler wiring only; fake inference and forced classification; no device action.'
        print(json.dumps(result, indent=2))
    finally:
        app.state.llm_queue.closing = True
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker


if __name__ == '__main__':
    asyncio.run(diagnose())
