"""Actual ASR/socket/admission and confirmation endpoints with fake inference."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
import app.main as runtime
from app.client_sessions import ClientSessions
from app.llm.conversation import ConversationGate
from test_shared_requests import server
from test_client_identity_server import Socket


async def until(predicate):
    for _ in range(200):
        if predicate(): return
        await asyncio.sleep(.005)
    raise AssertionError('Expected server event did not arrive')


@pytest.mark.asyncio
async def test_confirmation_runs_through_shared_queue_without_model_or_device_tools(server,monkeypatch):
    client,state=server
    state.client_sessions=ClientSessions();state.client_sessions.claim('kitchen','a')
    state.client_sessions.claim('office','b')
    state.conversation_gate=ConversationGate()
    infer=AsyncMock(side_effect=AssertionError('Mute commands must not call a model'))
    monkeypatch.setattr(runtime,'generate_llm_response',infer)
    for request,confirmation,seconds in [('Aria, mute yourself','yes',300),
                                         ('Aria, mets-toi en sourdine pendant dix minutes','oui',600)]:
        r=await client.post('/api/llm/process',json={'text':request,'session_id':'client:kitchen'})
        assert r.status_code==200 and r.json()['mute_status']=='confirmation_required'
        assert not state.client_sessions.is_muted('kitchen')
        r=await client.post('/api/llm/process',json={'text':confirmation,'session_id':'client:kitchen'})
        assert r.json()['mute_status']=='muted' and r.json()['tool_calls_made']==0
        assert 0 < state.client_sessions.mute_status('kitchen')['mute_remaining_s'] <= seconds
        assert not state.client_sessions.is_muted('office')
        assert (await client.post('/api/clients/kitchen/mute',json={'muted':False})).status_code==200
    infer.assert_not_awaited()
    assert (await client.post('/api/clients/no-such-client/mute',json={'muted':True})).status_code==404
    assert (await client.post('/api/clients/kitchen/mute',json={'muted':'false'})).status_code==422


@pytest.mark.asyncio
@pytest.mark.parametrize('end',['end','disconnect'])
async def test_muting_drops_partial_and_new_utterances_without_asr(server,monkeypatch,end):
    client,state=server; calls=[]
    state.asr=SimpleNamespace(transcribe=lambda pcm:calls.append(pcm) or 'hello')
    state.spk_recognizer=None;state.echo_guard=None
    monkeypatch.setattr(runtime,'_should_run_asr',lambda _:(True,{'seconds':1.,'speech_ratio':1.}))
    monkeypatch.setattr(runtime,'generate_llm_response',AsyncMock(return_value={'content':'Hello'}))
    ws=Socket('kitchen');task=asyncio.create_task(runtime.ws_asr(ws))
    try:
        await ws.ready.wait()
        ws.put({'type':'utterance_start'});ws.put(b'\0\0'*16000)
        await until(lambda:state.client_sessions.snapshot()[0]['transmitting'])
        await client.post('/api/clients/kitchen/mute',json={'muted':True})
        ws.put({'type':'utterance_end'})
        ws.put({'type':'utterance_start'});ws.put(b'\0\0'*16000);ws.put({'type':'utterance_end'})
        await asyncio.sleep(.03)
        assert calls==[]
        if end=='disconnect':
            ws.disconnect();await task;assert calls==[]
        else:
            await client.post('/api/clients/kitchen/mute',json={'muted':False})
            ws.put({'type':'utterance_start'});ws.put(b'\0\0'*16000);ws.put({'type':'utterance_end'})
            await until(lambda:any(m.get('status')=='completed' for m in ws.sent))
            assert len(calls)==1
    finally:
        if not task.done():ws.disconnect();await task


@pytest.mark.asyncio
async def test_mute_during_asr_invalidates_transcript_even_if_unmuted_before_asr_finishes(server,monkeypatch):
    client,state=server; entered=threading.Event();release=threading.Event()
    def transcribe(_):
        entered.set();release.wait(2);return 'hello'
    state.asr=SimpleNamespace(transcribe=transcribe);state.spk_recognizer=None;state.echo_guard=None
    monkeypatch.setattr(runtime,'_should_run_asr',lambda _:(True,{'seconds':1.,'speech_ratio':1.}))
    enqueue=AsyncMock(side_effect=AssertionError('Muted ASR result must not be admitted'))
    monkeypatch.setattr(runtime,'generate_llm_response',enqueue)
    ws=Socket('office');task=asyncio.create_task(runtime.ws_asr(ws))
    try:
        await ws.ready.wait()
        ws.put({'type':'utterance_start'});ws.put(b'\0\0'*16000);ws.put({'type':'utterance_end'})
        await until(entered.is_set)
        await client.post('/api/clients/office/mute',json={'muted':True})
        await client.post('/api/clients/office/mute',json={'muted':False})
        release.set();ws.disconnect();await task
        enqueue.assert_not_awaited()
        assert not any(m.get('status')=='accepted' for m in ws.sent)
    finally:
        release.set()
        if not task.done():ws.disconnect();await task
