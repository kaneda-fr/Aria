"""Read replies do not require a model, through both transports and history."""
from unittest.mock import AsyncMock, Mock
from types import SimpleNamespace

import pytest
import app.main as runtime
from test_shared_requests import server
from test_inventory_read_requests import snapshot


@pytest.mark.asyncio
async def test_read_path_without_model_preserves_history_and_language(server,snapshot,monkeypatch):
    client,state=server
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES','6')
    generator=AsyncMock(side_effect=AssertionError('read request must not invoke LLM'))
    monkeypatch.setattr(runtime,'generate_llm_response',generator)
    import app.llm.history_prefill as prefill
    schedule=Mock();monkeypatch.setattr(prefill,'schedule_prefill',schedule)
    state.llm_config=state.llm_client=None
    reader=AsyncMock(return_value={'success':True,'readings':[{
        'cmd_id':'1s','device_name':'Lampe Canapé','status':'ok','value':1,'metric':'state'}]})
    provider=SimpleNamespace(name='inventory',is_started=True,indexer=SimpleNamespace(get_snapshot=lambda:snapshot),read_resolved_state=reader)
    state.plugin_registry=SimpleNamespace(get_plugins_by_type=lambda _:[provider],get_plugin=lambda _:None)
    for i,text in enumerate(['Quelles commandes pour la lampe canapé dans le salon ?',
                             'Et son état ?', 'What is its state?', 'What are its capabilities?',
                             'Quel est son état ?']):
        if i%2:
            result=await runtime._enqueue_final_llm(runtime.app,text,'reads','unknown').result
        else:
            response=await client.post('/api/llm/process',json={'text':text,'session_id':'reads','speak_response':True})
            assert response.status_code==200,response.text
            result=response.json()
        assert result['success'] and result['response']
    assert len(state.llm_history['reads'])==6
    assert reader.await_count==3
    assert all(c.args==('1','state',None) for c in reader.await_args_list)
    assert [c.kwargs['language'] for c in state.speech_output.speak.await_args_list]==['fr','fr','en','en','fr']
    generator.assert_not_called();schedule.assert_not_called()
    result=await runtime._enqueue_final_llm(runtime.app,'What is its state?','other','unknown').result
    assert 'specify' in result['response'].lower()
    result=await runtime._enqueue_final_llm(runtime.app,'What is its state?','reads','other-person').result
    assert 'specify' in result['response'].lower()
    assert reader.await_count==3


@pytest.mark.asyncio
async def test_tool_opt_out_retains_conversation_path(server,snapshot,monkeypatch):
    client,state=server
    generator=AsyncMock(return_value={'content':'Tools are disabled.'})
    monkeypatch.setattr(runtime,'generate_llm_response',generator)
    provider=SimpleNamespace(name='inventory',is_started=True,indexer=SimpleNamespace(get_snapshot=lambda:snapshot),read_resolved_state=AsyncMock())
    state.plugin_registry=SimpleNamespace(get_plugins_by_type=lambda _:[provider],get_plugin=lambda _:None)
    result=await client.post('/api/llm/process',json={'text':'What is the state of Lampe Bureau?','use_tools':False,'speak_response':False})
    assert result.status_code==200 and result.json()['response']=='Tools are disabled.'
    generator.assert_awaited_once();provider.read_resolved_state.assert_not_called()


@pytest.mark.asyncio
async def test_english_intent_language_outranks_french_inventory_labels(server,snapshot,monkeypatch):
    client,state=server
    monkeypatch.setattr(runtime,'detect_language',lambda _: 'fr')
    generator=AsyncMock(side_effect=AssertionError('no inference'))
    monkeypatch.setattr(runtime,'generate_llm_response',generator)
    provider=SimpleNamespace(name='inventory',is_started=True,indexer=SimpleNamespace(get_snapshot=lambda:snapshot))
    state.plugin_registry=SimpleNamespace(get_plugins_by_type=lambda _:[provider],get_plugin=lambda _:None)
    response=await client.post('/api/llm/process',json={'text':'What controls are available for Lampe Canapé in Salon?','session_id':'language','speak_response':True})
    assert response.status_code==200
    assert response.json()['response'].startswith('Available controls')
    assert state.speech_output.speak.await_args.kwargs['language']=='en'
    generator.assert_not_called()
