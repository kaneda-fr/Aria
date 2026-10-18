"""Ollama residency survives streaming, warm-up, prefill, and tool requests."""
from dataclasses import replace
import json

import httpx
import pytest

from app.llm.ollama_client import (
    OllamaConfig, LlamaCppConfig, load_ollama_config, generate_llm_response,
)
from scripts.e2e.live_warmup import warmup


@pytest.mark.parametrize('value,expected', [(None,-1), ('-1',-1), ('0',0), ('300',300)])
def test_residency_configuration(monkeypatch,value,expected):
    monkeypatch.delenv('ARIA_LLM_KEEP_ALIVE_S',raising=False)
    if value is not None:
        monkeypatch.setenv('ARIA_LLM_KEEP_ALIVE_S',value)
    assert load_ollama_config().keep_alive_s == expected


@pytest.mark.parametrize('value', ['-2','5m','nan',''])
def test_invalid_residency_fails_at_configuration(monkeypatch,value):
    monkeypatch.setenv('ARIA_LLM_KEEP_ALIVE_S',value)
    with pytest.raises(ValueError):
        load_ollama_config()


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False,True])
@pytest.mark.parametrize('keep_alive', [-1,0,300])
async def test_all_ollama_requests_preserve_residency_and_tools(stream,keep_alive):
    bodies=[]
    call={'function':{'name':'find_devices','arguments':{'query':'lamp'}}}
    def handle(request):
        body=json.loads(request.content);bodies.append(body)
        response={'message':{'content':'','tool_calls':[call]},'done':True}
        return httpx.Response(200,content=json.dumps(response)+'\n',
                              headers={'content-type':'application/x-ndjson' if stream else 'application/json'})
    cfg=OllamaConfig('ollama','http://local','fixture',stream=stream,keep_alive_s=keep_alive)
    messages=[{'role':'user','content':'find the lamp'}]
    tools=[{'type':'function','function':{'name':'find_devices'}}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result=await generate_llm_response(text='',messages=messages,config=cfg,client=client,tools=tools)
        # The same dataclass replacement used by prefill must retain the policy.
        await generate_llm_response(text='',messages=messages,config=replace(cfg,num_predict=1,stream=False),client=client,tools=tools)
        primed=await warmup(cfg,client,'system',tools)
    assert result['tool_calls']==[call] and primed['discarded_tool_calls']==1
    assert len(bodies)==3
    assert all(b['keep_alive']==keep_alive and b['tools']==tools for b in bodies)
    assert bodies[0]['messages']==messages
    assert 'options' not in bodies[0]  # Residency is not a generation/sampling option.
    assert bodies[1]['options']['num_predict']==1


@pytest.mark.asyncio
async def test_llamacpp_does_not_receive_ollama_residency():
    bodies=[]
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':'OK'}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        response=await generate_llm_response(text='Hello',config=LlamaCppConfig('llamacpp','http://local/v1'),client=client)
    assert response['content']=='OK' and 'keep_alive' not in bodies[0]
