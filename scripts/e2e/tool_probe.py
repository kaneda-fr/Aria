"""Real-model tool diagnostic; run on Apollo inside a loopback-only namespace.

Uses the shared HTTP/voice worker, real inventory matching and actions payload
validation. Only MQTT publish is replaced; no broker or physical device is used.
This intentionally does not start ASR/TTS or claim an acoustic end-to-end pass.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import socket
from pathlib import Path
import time
from types import SimpleNamespace

import httpx

import app.main as runtime
import app.llm.tools as tooling
from app.llm.ollama_client import load_ollama_config, SYSTEM_PROMPT
from app.llm.request_queue import RequestQueue
from app.plugins.actions.provider import ActionsProvider
from app.plugins.inventory.models import CommandRecord, DeviceRecord
from app.plugins.inventory.provider import InventoryProvider


CASES = [
    ('on-en', 'Turn on the kitchen light.', [('201', {})]),
    ('on-fr', 'Allume la lumière de la cuisine.', [('201', {})]),
    ('off-en', 'Turn off the kitchen light.', [('202', {})]),
    ('off-fr', 'Éteins la lumière de la cuisine.', [('202', {})]),
    ('dim-en', 'Set the kitchen light to 30 percent.', [('203', {'slider': 30})]),
    ('dim-fr', 'Règle la lumière de la cuisine à 30 pour cent.', [('203', {'slider': 30})]),
    ('missing-en', 'Turn on the garage light.', []),
    ('missing-fr', 'Allume la lumière du garage.', []),
    ('ambiguous-en', 'Turn on the light.', []),
    ('ambiguous-fr', 'Allume la lumière.', []),
    ('chat-en', 'What is the capital of France?', []),
    ('chat-fr', 'Quelle est la capitale de la France ?', []),
]


def inventory_fixture() -> InventoryProvider:
    inventory = InventoryProvider('inventory', {})
    for room, base in [('cuisine', 200), ('chambre', 300)]:
        commands = []
        for suffix, capability in [(1, 'light_on'), (2, 'light_off'), (3, 'light_slider')]:
            slider = suffix == 3
            commands.append(CommandRecord(
                cmd_id=str(base + suffix), device_id=room, type='action',
                subType='slider' if slider else 'other', capability=capability,
                unit='%' if slider else None, range=(0, 100) if slider else None,
                tags=set(), execution={'args': {
                    'value_required': slider, 'value_type': 'int' if slider else None,
                    'range': [0, 100] if slider else None,
                    'payload_template': 'slider' if slider else 'other'}},
                name=capability))
        inventory.indexer.add_device(DeviceRecord(
            source='fixture', eq_id=room, name=f'Lampe {room}', room=room,
            domain='light', enabled=True, visible=True, commands=commands,
            tags=set(), seen_at=0))
    inventory._mark_started()
    return inventory


async def main() -> None:
    run = Path(os.environ['ARIA_E2E_RUN'])
    # Defense in depth: this diagnostic must not have a LAN-capable interface.
    assert {name for _, name in socket.if_nameindex()} == {'lo'}, 'Requires an isolated network namespace'
    config = load_ollama_config()
    assert config.url == 'http://127.0.0.1:11435'
    os.environ['ARIA_LLM_HISTORY_MAX_MESSAGES'] = '0'
    inventory = inventory_fixture()
    published, models, calls, lifecycle = [], [], [], []

    def publish(topic, payload, qos):
        published.append({'cmd_id': topic.rsplit('/', 1)[-1], 'payload': json.loads(payload)})
        return SimpleNamespace(rc=0, mid=len(published))

    actions = ActionsProvider('actions', {'root_topic': 'fixture'})
    actions._client = SimpleNamespace(publish=publish)
    actions._connected = True
    actions._mark_started()
    original_execute = tooling.execute_tool

    async def observe_tool(**kwargs):
        result = await original_execute(**kwargs)
        calls.append({'name': kwargs['tool_name'], 'arguments': kwargs['tool_args'], 'result': result})
        return result

    async def observe_model(response):
        if response.request.url.path != '/api/chat':
            return
        await response.aread()
        data = response.json()
        models.append({'request': json.loads(response.request.content), 'response': data})

    tooling.execute_tool = observe_tool
    state = runtime.app.state
    state.llm_queue = RequestQueue(on_event=lambda event: lifecycle.append(event))
    state.llm_config = config
    state.router = None
    state.speech_output = None
    state.llm_history = {}
    state.llm_history_lock = asyncio.Lock()
    state.plugin_registry = SimpleNamespace(
        get_plugins_by_type=lambda _: [inventory], get_plugin=lambda _: actions)
    report = {'scope': 'real model/shared worker/real inventory and payloads; captured publish; no ASR/TTS',
              'system_prompt': SYSTEM_PROMPT, 'model': config.model,
              'num_predict': config.num_predict, 'temperature': config.temperature,
              'top_p': config.top_p, 'history': 0,
              'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                  'app/main.py', 'app/llm/tools.py', 'app/llm/ollama_client.py',
                  'app/plugins/inventory/provider.py', 'app/plugins/actions/provider.py')}, 'cases': []}
    async with httpx.AsyncClient(timeout=config.timeout_s, event_hooks={'response': [observe_model]}) as model_client:
        state.llm_client = model_client
        report['models'] = (await model_client.get(config.url + '/api/tags')).json()
        worker = asyncio.create_task(runtime._llm_worker(runtime.app))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app), base_url='http://aria') as client:
                # Twelve HTTP cases plus two paired finalized-voice handler cases.
                for source, cases in [('api', CASES), ('voice', CASES[:2])]:
                    for case_id, text, expected in cases:
                        published.clear(); models.clear(); calls.clear(); lifecycle.clear()
                        started = time.monotonic()
                        if source == 'api':
                            response = await client.post('/api/llm/process', json={
                                'text': text, 'session_id': case_id, 'speak_response': False})
                            result = response.json()
                        else:
                            job = runtime._enqueue_final_llm(runtime.app, text, case_id, 'unknown')
                            result = await job.result
                        actual = [(p['cmd_id'], p['payload']) for p in published]
                        entry = {'id': case_id, 'source': source, 'text': text,
                                 'expected_publishes': expected, 'published': list(published),
                                 'publish_match': actual == expected, 'result': result,
                                 'seconds': time.monotonic() - started, 'tools': list(calls),
                                 'models': list(models), 'lifecycle': list(lifecycle)}
                        report['cases'].append(entry)
                        (run / 'tool-report.json').write_text(json.dumps(report, indent=2))
                        print(json.dumps({k: entry[k] for k in ('id', 'source', 'publish_match', 'published', 'result', 'seconds')}), flush=True)
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            tooling.execute_tool = original_execute


if __name__ == '__main__':
    asyncio.run(main())
