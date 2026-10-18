"""Existing /ws/asr compatibility: no new capabilities or acknowledgment."""
import asyncio
import ast
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from app.audio.ringbuffer import RingBuffer


class ClientSignalsTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_is_accepted_without_server_ack(self):
        messages, captured = await self.session('utterance_end')
        self.assertEqual(messages, [])
        self.assertEqual(captured, [b"\x01\x00" * 320])

    async def test_stop_retains_existing_finalize(self):
        messages, captured = await self.session('stop')
        self.assertEqual(messages, [])
        self.assertEqual(captured, [b"\x01\x00" * 320])

    async def test_disconnect_retains_existing_finalize(self):
        _, captured = await self.session(None)
        self.assertEqual(captured, [b"\x01\x00" * 320])

    async def test_explicit_end_is_not_repeated_on_disconnect(self):
        _, captured = await self.session('utterance_end')
        self.assertEqual(len(captured), 1)

    async def session(self, ending):
        tree = ast.parse(Path('app/main.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'ws_asr')
        node.decorator_list = []
        node.returns = None
        for argument in node.args.args: argument.annotation = None
        validator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_valid_start')
        app = SimpleNamespace(state=SimpleNamespace(asr=None, spk_recognizer=None, echo_guard=None))
        captured = []
        class Vad:
            def __init__(self, _): pass
            def reset(self): pass
        namespace = dict(asyncio=asyncio, contextlib=contextlib, json=json, Any=object,
                         time=__import__('time'), os=SimpleNamespace(environ={}),
                         uuid4=__import__('uuid').uuid4, log=SimpleNamespace(info=lambda *a, **k: None),
                         RingBuffer=RingBuffer, VoiceActivityDetector=Vad, VadConfig=lambda **k: k,
                         app=app, _safe_json=json.loads,
                         _should_run_asr=lambda pcm: (captured.append(pcm) or False,
                                                     {'seconds': 0., 'speech_ratio': 0.}),
                         _filter_transcript=lambda text, **_: text)
        exec(compile(ast.Module(body=[validator, node], type_ignores=[]), 'app/main.py', 'exec'), namespace)
        def text(kind, **values): return {'text': json.dumps({'type': kind, **values})}
        messages = [text('start', sample_rate=16000, channels=1, format='pcm_s16le',
                         chunk_ms=20, client_vad=True),
                    text('utterance_start'), {'bytes': b"\x01\x00" * 320}]
        if ending: messages.append(text(ending))
        messages.append({'type': 'websocket.disconnect'})
        sent = []
        class Socket:
            async def accept(self): pass
            async def close(self, **_): pass
            async def receive(self):
                await asyncio.sleep(0)
                return messages.pop(0)
            async def send_json(self, frame): sent.append(frame)
        await namespace['ws_asr'](Socket())
        return sent, captured
