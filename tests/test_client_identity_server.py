"""Identity/status integration through actual shared server handlers."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import app.main as runtime
from test_shared_requests import server


class Socket:
    def __init__(self, name):
        self.accept=AsyncMock();self.close=AsyncMock();self.inbox=asyncio.Queue()
        self.sent=[];self.ready=asyncio.Event()
        self.put(dict(type='start',sample_rate=16000,channels=1,format='pcm_s16le',
                      chunk_ms=20,client_vad=True,client_id=name))
    def put(self, value):
        self.inbox.put_nowait({'bytes':value} if isinstance(value,bytes) else {'text':json.dumps(value)})
    async def receive(self):return await self.inbox.get()
    async def send_json(self, message):
        self.sent.append(message)
        if message['type']=='session_started':self.ready.set()
    def disconnect(self):self.inbox.put_nowait({'type':'websocket.disconnect'})


@pytest.mark.asyncio
async def test_clients_endpoint_reports_actual_queue_work(server,monkeypatch):
    client,state=server
    state.asr=None;state.spk_recognizer=None;state.echo_guard=None
    entered,release=asyncio.Event(),asyncio.Event()
    async def infer(**_):
        entered.set();await release.wait();return {'content':'Hello'}
    monkeypatch.setattr(runtime,'generate_llm_response',infer)
    ws=Socket('kitchen');handler=asyncio.create_task(runtime.ws_asr(ws))
    try:
        await asyncio.wait_for(ws.ready.wait(),2)
        task=asyncio.create_task(client.post('/api/llm/process',json={
            'text':'Hello','session_id':'client:kitchen','speak_response':False,'use_tools':False}))
        await asyncio.wait_for(entered.wait(),2)
        row=(await client.get('/api/clients')).json()['clients'][0]
        assert row['client_id']=='kitchen' and row['connected'] and row['processing']
        assert not row['transmitting']
        release.set();assert (await task).status_code==200
        assert not (await client.get('/api/clients')).json()['clients'][0]['processing']
    finally:
        release.set();ws.disconnect();await handler
    assert not (await client.get('/api/clients')).json()['clients'][0]['connected']


@pytest.mark.asyncio
async def test_speaker_score_is_tied_to_exact_voice_request(server,monkeypatch,caplog):
    client,state=server
    caplog.set_level('INFO')
    state.asr=SimpleNamespace(transcribe=lambda _:'Bonjour')
    state.echo_guard=None
    state.spk_recognizer=SimpleNamespace(enabled=True,self_names=(),self_min_score=.75,
        identify=lambda *a,**k:SimpleNamespace(user='person',name='person_fr',score=.42,
                                             known=False,second_best=.2))
    monkeypatch.setattr(runtime,'_should_run_asr',lambda _:(True,{'seconds':1.,'speech_ratio':1.}))
    monkeypatch.setattr(runtime,'generate_llm_response',AsyncMock(return_value={'content':'Bonjour'}))
    ws=Socket('office');handler=asyncio.create_task(runtime.ws_asr(ws))
    try:
        await asyncio.wait_for(ws.ready.wait(),2)
        ws.put({'type':'utterance_start'});ws.put(b'\0\0'*16000);ws.put({'type':'utterance_end'})
        for _ in range(200):
            if any(m.get('status')=='completed' for m in ws.sent):break
            await asyncio.sleep(.01)
        completed=next(m for m in ws.sent if m.get('status')=='completed')
        record=next(r.fields for r in caplog.records if r.message=='ARIA.SPK.Request')
        assert record['request_id']==completed['request_id']
        assert record['client_id']=='office' and record['user']=='person'
        assert record['score']==.42 and record['known'] is False
    finally:
        ws.disconnect();await handler
