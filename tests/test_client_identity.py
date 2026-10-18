"""Connection identity, conversation continuity and live status without models."""
import ast
import asyncio
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from app.client_sessions import ClientSessions, validate_client_id
from app.audio.ringbuffer import RingBuffer
from app.llm.conversation import ConversationGate


@pytest.mark.parametrize('bad', ['', None, 123, 'A', 'bad name', '../x', 'x'*65])
def test_identity_rejects_invalid_names(bad):
    with pytest.raises(ValueError): validate_client_id({'client_id': bad})


def test_identity_is_name_and_absence_is_legacy():
    assert validate_client_id({}) is None
    assert validate_client_id({'client_id': 'kitchen-esp32'}) == 'kitchen-esp32'


def test_registry_reconnect_duplicate_bound_and_status():
    registry = ClientSessions(limit=2)
    session = registry.claim('kitchen', 'first')
    with pytest.raises(ValueError, match='in_use'): registry.claim('kitchen', 'duplicate')
    registry.release('kitchen', 'duplicate')
    assert registry.active['kitchen'] == 'first'
    registry.audio('kitchen', 'first')
    registry.request(dict(session_id=session, request_id='r1', status='running'))
    assert registry.snapshot()[0]['processing'] and registry.snapshot()[0]['transmitting']
    registry.release('kitchen', 'first')
    assert not registry.snapshot()[0]['connected']
    assert registry.claim('kitchen', 'second') == session
    registry.request(dict(session_id=session, request_id='r1', status='completed'))
    assert not registry.snapshot()[0]['processing']
    registry.claim('bedroom', 'third')
    with pytest.raises(ValueError, match='limit'): registry.claim('office', 'fourth')
    registry.release('kitchen', 'second')
    registry.claim('office', 'fourth')
    assert len(registry.snapshot()) == 2


def test_reconnect_preserves_wake_only_until_existing_expiry():
    clock = [0.]
    gate = ConversationGate(clock=lambda: clock[0], idle_s=30)
    registry = ClientSessions()
    key = registry.claim('kitchen', 'a')
    turn = gate.begin(key, 'Aria, hello', 0.)
    turn.session.busy = False
    registry.release('kitchen', 'a')
    assert gate.begin(registry.claim('kitchen', 'b'), 'How are you?', 2.).kind == 'followup'
    turn.session.busy = False
    assert gate.begin(key, 'How are you?', 31.).kind == 'ignored'
    assert gate.begin('client:bedroom', 'How are you?', 2.).kind == 'ignored'


@pytest.mark.asyncio
async def test_real_handler_reconnect_duplicate_and_legacy_cleanup():
    tree = ast.parse(Path('app/main.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'ws_asr')
    node.decorator_list = []; node.returns = None
    for argument in node.args.args: argument.annotation = None
    validator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_valid_start')
    state = SimpleNamespace(asr=None, spk_recognizer=None, echo_guard=None,
                            llm_history={'client:kitchen': ['retained']}, llm_history_lock=asyncio.Lock())
    namespace = dict(asyncio=asyncio, contextlib=contextlib, json=json, Any=object,
                     time=__import__('time'), os=SimpleNamespace(environ={}), uuid4=uuid4,
                     log=SimpleNamespace(info=lambda *a, **k: None), RingBuffer=RingBuffer,
                     VoiceActivityDetector=lambda _: SimpleNamespace(reset=lambda: None),
                     VadConfig=lambda **k: k, app=SimpleNamespace(state=state), _safe_json=json.loads)
    exec(compile(ast.Module(body=[validator, node], type_ignores=[]), 'app/main.py', 'exec'), namespace)
    class Socket:
        def __init__(self, name=None):
            self.accept=AsyncMock(); self.close=AsyncMock(); self.sent=[]; self.queue=asyncio.Queue()
            payload=dict(type='start',sample_rate=16000,channels=1,format='pcm_s16le',chunk_ms=20)
            if name is not None: payload['client_id']=name
            self.queue.put_nowait({'text':json.dumps(payload)})
        async def receive(self): return await self.queue.get()
        async def send_json(self, message): self.sent.append(message)
        def disconnect(self): self.queue.put_nowait({'type':'websocket.disconnect'})
    async def connected(socket):
        task=asyncio.create_task(namespace['ws_asr'](socket))
        for _ in range(50):
            if socket.sent: break
            await asyncio.sleep(.001)
        return task
    a=Socket('kitchen'); task=await connected(a)
    assert a.sent[0]['client_id']=='kitchen' and a.sent[0]['session_id']=='client:kitchen'
    duplicate=Socket('kitchen'); await namespace['ws_asr'](duplicate)
    assert duplicate.sent[0]['error']=='client_id_in_use'
    assert state.client_sessions.snapshot()[0]['connected']
    a.disconnect(); await task
    assert state.llm_history['client:kitchen']==['retained']
    b=Socket('kitchen'); task=await connected(b)
    assert b.sent[0]['session_id']==a.sent[0]['session_id']
    assert b.sent[0]['connection_id']!=a.sent[0]['connection_id']
    b.disconnect(); await task
    legacy=Socket(); legacy.disconnect(); await namespace['ws_asr'](legacy)
    assert legacy.sent==[]


def test_mac_name_is_stable_and_override_is_validated(monkeypatch):
    tree=ast.parse(Path('client/macos/mic_stream.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='client_name')
    import os, re
    ns=dict(os=os,re=re,socket=SimpleNamespace(gethostname=lambda:'Dev-Mac.local'))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'client-name','exec'),ns)
    monkeypatch.delenv('ARIA_CLIENT_ID',raising=False)
    assert ns['client_name']()==ns['client_name']()=='aria-mac-dev-mac'
    monkeypatch.setenv('ARIA_CLIENT_ID','office-mac')
    assert ns['client_name']()=='office-mac'
    monkeypatch.setenv('ARIA_CLIENT_ID','bad/name')
    with pytest.raises(ValueError): ns['client_name']()
