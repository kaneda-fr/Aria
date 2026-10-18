import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import app.main as runtime
from app.llm.conversation import ConversationGate
from app.speaker_recognition.recognizer import SpeakerResult
from test_shared_requests import server


@pytest.mark.asyncio
@pytest.mark.parametrize('score,known,echo,rejected', [
    (.126,False,False,False), (.70,True,False,False),
    (.80,True,False,True), (.126,False,True,True),
])
async def test_real_voice_path_confident_self_and_echo_still_suppress(
        server,monkeypatch,score,known,echo,rejected):
    _,state=server
    state.conversation_gate=ConversationGate()
    state.asr=SimpleNamespace(transcribe=lambda _: 'Hello Aria')
    state.spk_recognizer=SimpleNamespace(enabled=True,self_min_score=.75,self_names=('aria_en',),
        identify=lambda *a,**kw: SpeakerResult('aria_en',score,known,'aria',.05))
    state.echo_guard=SimpleNamespace(enabled=True,should_suppress=lambda *a:(echo,{}))
    monkeypatch.setattr(runtime,'_should_run_asr',lambda _:(True,dict(seconds=1.,rms=.1,speech_ratio=1.)))
    messages=asyncio.Queue()
    for data in [dict(type='start',sample_rate=16000,channels=1,format='pcm_s16le',chunk_ms=40,client_vad=True),
                 dict(type='utterance_start'),b'\0\0'*16000,dict(type='utterance_end')]:
        messages.put_nowait({'type':'websocket.receive',**({'bytes':data} if isinstance(data,bytes) else {'text':json.dumps(data)})})
    async def receive():
        if messages.empty():return {'type':'websocket.disconnect'}
        return await messages.get()
    jobs=[]
    original=runtime._enqueue_final_llm
    def enqueue(*a,**kw):
        job=original(*a,**kw);jobs.append(job);return job
    monkeypatch.setattr(runtime,'_enqueue_final_llm',enqueue)
    ws=SimpleNamespace(accept=AsyncMock(),close=AsyncMock(),receive=receive,send_json=AsyncMock())
    await runtime.ws_asr(ws)
    await asyncio.wait_for(state.llm_queue.pending.join(),2)
    if rejected:
        assert not jobs and not state.conversation_gate.sessions
        state.speech_output.speak.assert_not_awaited()
    else:
        job,=jobs
        assert job.speaker_user=='unknown' and job.text=='Hello Aria'
        assert job.result.result()['response']=='Yes?'


@pytest.mark.parametrize('raw,canonical', [
    ('Bonjour Ari.','Bonjour Aria.'), ('Hello area.','Hello Aria.'),
    ('Aria, turn it off','Aria, turn it off'), ('Ariya','Aria'), ('Arya!','Aria!'),
    ('Ari, what time is it?','Aria, what time is it?'),
    ('Hey area, what time is it?','Hey Aria, what time is it?'),
    ('Goodbye area.','Goodbye Aria'), ('Merci, au revoir Ari!','Merci, au revoir Aria'),
])
def test_spoken_vocative_only(raw,canonical):
    from app.llm.conversation import normalize_spoken_vocative
    assert normalize_spoken_vocative(raw)==canonical


@pytest.mark.parametrize('text', ['area', 'area of the kitchen', 'Hello area code 123',
    'What is the area?', 'Who is Arya?', 'Bonjour Marie', 'Maria, turn it off',
    'Ariana', 'Ohio', "À l'eau", 'You would buy area.', 'Explain goodbye area',
    'Aria, what is the area of the room?'])
def test_spoken_normalization_does_not_rewrite_unrelated_words(text):
    from app.llm.conversation import normalize_spoken_vocative
    assert normalize_spoken_vocative(text)==text


@pytest.mark.asyncio
async def test_shared_gate_only_normalizes_voice_and_retains_raw_input(server):
    _,state=server
    state.conversation_gate=ConversationGate()
    async def request(text,source):
        job=state.llm_queue.submit(text=text,session_id=source,speaker_user='unknown',
                                  speak_response=True,use_tools=True,source=source)
        return await job.result
    assert (await request('Hello area.','api'))['ignored']
    result=await request('Hello area.','voice')
    assert result['text']=='Hello area.' and result['response']=='Yes?'
    assert result['conversation_action']=='activated'
    assert (await request('Goodbye area.','voice'))['conversation_action']=='closed'
