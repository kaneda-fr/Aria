"""Retry lifecycle with real client tasks, fake sockets and fake audio capture.

No microphone, network, playback, models or devices are accessed.
"""
import asyncio
import contextlib
import io
import json
import signal
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from client.macos import mic_stream as mic


class ClientReconnectTests(unittest.IsolatedAsyncioTestCase):
    async def exercise_disconnect(self, *, rejected=False, connect_failure=False):
        disconnected = asyncio.Event()
        handlers = {}
        attempts = []

        class Socket:
            def __init__(self):
                self.reported_rejection = False

            async def send(self, message):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                if rejected and not self.reported_rejection:
                    self.reported_rejection = True
                    return json.dumps({'type': 'session_error', 'error': 'client_id_in_use'})
                if rejected:
                    raise RuntimeError('mock policy close')
                raise StopAsyncIteration

        class Connection:
            async def __aenter__(self):
                attempts.append(asyncio.get_running_loop().time())
                if connect_failure:
                    disconnected.set()
                    raise OSError('mock connection failure')
                return Socket()

            async def __aexit__(self, *args):
                disconnected.set()

        loop = asyncio.get_running_loop()
        fake_audio = SimpleNamespace(
            default=SimpleNamespace(device=(0, 0)),
            query_devices=Mock(return_value={'default_samplerate': 48000, 'name': 'fake input'}),
            InputStream=Mock(return_value=contextlib.nullcontext()),
        )
        output = io.StringIO()
        with patch.dict(mic.os.environ, {'ARIA_CLIENT_ID': 'test-mac', 'ARIA_CLIENT_VAD': '0'}, clear=True), \
             patch.object(mic, 'sd', fake_audio), \
             patch.object(mic.websockets, 'connect', return_value=Connection()), \
             patch.object(loop, 'add_signal_handler', side_effect=lambda sig, callback: handlers.update({sig: callback})), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            client = asyncio.create_task(mic.run_client('ws://127.0.0.1/ws/asr', mic.AriaAudioContract(), device=None))
            try:
                await asyncio.wait_for(disconnected.wait(), timeout=1)
                # A closed/rejected session used to reopen capture every 100 ms.
                await asyncio.sleep(.25)
                self.assertEqual(len(attempts), 1, 'session retried before the five-second delay')
                self.assertIn('retrying in 5s', output.getvalue())
                self.assertEqual(fake_audio.InputStream.call_count, 0 if connect_failure else 1)
                handlers[signal.SIGINT]()
                try:
                    result = await asyncio.wait_for(client, timeout=.25)
                except asyncio.TimeoutError:
                    self.fail('stop did not interrupt the retry wait')
                self.assertEqual(result, 0)
                self.assertEqual(len(attempts), 1)
            finally:
                client.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await client

    async def test_clean_close_delays_reconnect(self):
        await self.exercise_disconnect()

    async def test_rejected_session_delays_reconnect(self):
        await self.exercise_disconnect(rejected=True)

    async def test_stop_interrupts_connection_failure_retry(self):
        await self.exercise_disconnect(connect_failure=True)
