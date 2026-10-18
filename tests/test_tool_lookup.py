"""Native template repair and truthful command-vs-device grounding."""
from dataclasses import replace
import json

import httpx
import pytest

from app.llm.ollama_client import OllamaConfig, LlamaCppConfig
from app.llm.ollama_template import prepare_tool_template
from app.llm.tools import format_tool_result_for_llm


@pytest.mark.asyncio
@pytest.mark.parametrize('cached', [False, True])
async def test_template_alias_preserves_weights_config_and_tool_schema(cached):
    template = '{{ range .Tools }}{"type": "function", "function": {{ .Function }}}{{ end }} {"name": <function-name>}'
    fixed = '{{ range .Tools }}{{ . }}{{ end }} {"name": "<function-name>"}'
    calls = []
    def handle(request):
        body = json.loads(request.content); calls.append((request.url.path, body))
        if body.get('model') == 'original':
            return httpx.Response(200, json={'template': template, 'modelfile': 'FROM existing-weights'})
        if request.url.path == '/api/create':
            assert body['from'] == 'original' and body['template'] == fixed
            assert set(body) == {'model', 'from', 'template', 'stream'}
            return httpx.Response(200, json={'status': 'success'})
        return httpx.Response(200, json={'template': fixed}) if cached else httpx.Response(404)
    config = OllamaConfig('ollama', 'http://local', 'original', stream=True, num_predict=80)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        prepared = await prepare_tool_template(config, client)
    assert prepared.model.startswith('aria-tools-')
    assert replace(prepared, model=config.model) == config
    assert len(calls) == (2 if cached else 3)
    assert all(path not in ('/api/pull', '/api/chat', '/api/generate') for path, _ in calls)


@pytest.mark.asyncio
async def test_unaffected_template_and_other_backend_are_unchanged():
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={'template': '{{ .Tools }}'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        for cfg in [OllamaConfig('ollama', 'http://local', 'model'), LlamaCppConfig('llamacpp', 'http://local')]:
            assert await prepare_tool_template(cfg, client) is cfg
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['create_error', 'collision', 'missing_model'])
async def test_template_preparation_fails_closed_without_pull(failure):
    def handle(request):
        body = json.loads(request.content)
        assert request.url.path != '/api/pull'
        if body.get('model') == 'original':
            return httpx.Response(404) if failure == 'missing_model' else httpx.Response(200, json={
                'template': '{"type": "function", "function": {{ .Function }}}'})
        if request.url.path == '/api/create':
            return httpx.Response(200, json={'error': 'disk full'})
        return httpx.Response(200, json={'template': 'wrong'}) if failure == 'collision' else httpx.Response(404)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises((RuntimeError, httpx.HTTPStatusError)):
            await prepare_tool_template(OllamaConfig('ollama', 'http://local', 'original'), client)


@pytest.mark.parametrize('language', ['fr', 'en'])
def test_lookup_groups_commands_by_identity_and_keeps_truncation_explicit(language):
    candidates = [dict(device_id=device, device_name='Same name', room='Same room',
                       cmd_id=device+cap, capability='light_'+cap)
                  for device, cap in [('one', 'on'), ('one', 'off'), ('two', 'on')]]
    text = format_tool_result_for_llm({'success': True, 'candidates': candidates, 'total_found': 9}, language=language)
    data = json.loads(text.split('\n')[0])
    assert data['shown_devices'] == 2 and data['shown_commands'] == 3 and data['matching_commands'] == 9
    assert len(data['devices'][0]['commands']) == 2
    assert data['current_state'] == 'unknown'
    assert ('français' in text) == (language == 'fr')
    assert ('Aucune action exécutée' in text) == (language == 'fr')
