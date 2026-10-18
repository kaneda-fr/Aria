from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.llm.ollama_client import OllamaConfig
from scripts.e2e import live_warmup


@pytest.mark.asyncio
async def test_warmup_discards_calls_and_preserves_operational_config(monkeypatch):
    config = OllamaConfig(provider='ollama', url='http://local', model='fixture',
                          timeout_s=10, num_predict=80, temperature=0, top_p=.85, stream=False)
    generate = AsyncMock(return_value={'content': 'DO NOT SPEAK',
                                      'tool_calls': [{'function': {'name': 'execute_command'}}]})
    monkeypatch.setattr(live_warmup, 'generate_llm_response', generate)
    client, tools = object(), [{'function': {'name': 'execute_command'}}]
    result = await live_warmup.warmup(config, client, 'actual system prompt', tools)
    args = generate.call_args.kwargs
    assert args['tools'] is tools and args['client'] is client
    assert args['messages'][0]['content'] == 'actual system prompt'
    assert args['messages'][1]['content'].startswith('[speaker=unknown] ')
    assert args['config'].num_predict == 8 and config.num_predict == 80
    assert result['discarded_tool_calls'] == 1
    assert 'content' not in result and 'tool_calls' not in result


@pytest.mark.asyncio
async def test_warmup_failure_propagates(monkeypatch):
    monkeypatch.setattr(live_warmup, 'generate_llm_response', AsyncMock(side_effect=RuntimeError('offline error')))
    with pytest.raises(RuntimeError, match='offline error'):
        await live_warmup.warmup(SimpleNamespace(), object(), 'system', [], with_tools=False)
