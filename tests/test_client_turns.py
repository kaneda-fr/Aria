"""Endpoint policy tests: deterministic VAD/resampling, real client sender.

No microphone/device IO; Apollo baseline separately covers real WebRTC + soxr.
"""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from client.macos import mic_stream as mic


class ClientTurnTests(unittest.IsolatedAsyncioTestCase):
    async def send_frames(self, segments, silence_ms='800'):
        messages = []
        async def send(message):
            messages.append(message)
        queue = asyncio.Queue(maxsize=4)
        stop = asyncio.Event()
        fake_vad = SimpleNamespace(is_speech=lambda data, _: any(data))
        identity = SimpleNamespace(resample_chunk=lambda block, **_: block)
        env = {'ARIA_CLIENT_AGC': '0', 'ARIA_CLIENT_GAIN': '1',
               'ARIA_CLIENT_VAD_START_FRAMES': '3', 'ARIA_CLIENT_VAD_PRE_MS': '200',
               'ARIA_CLIENT_VAD_MIN_RMS': '0', 'ARIA_CLIENT_VAD_SNR_DB': '0',
               'ARIA_CLIENT_VAD_SILENCE_MS': silence_ms or '800'}
        with patch.dict(mic.os.environ, env, clear=True), \
             patch.object(mic.soxr, 'ResampleStream', return_value=identity), \
             patch.object(mic, 'webrtcvad', SimpleNamespace(Vad=lambda _: fake_vad)):
            if silence_ms is None:
                del mic.os.environ['ARIA_CLIENT_VAD_SILENCE_MS']
            sender = asyncio.create_task(mic._send_loop(
                SimpleNamespace(send=send), queue, stop, in_sample_rate=16000,
                contract=mic.AriaAudioContract(), stats=mic.ClientStats(), client_vad_enabled=True))
            try:
                for speech, duration_ms in segments:
                    for _ in range(duration_ms // 20):
                        await asyncio.wait_for(queue.put(np.full(320, .1 if speech else 0, dtype=np.float32)), 2)
                stop.set()
                await asyncio.wait_for(sender, 2)
            finally:
                sender.cancel()
        return [json.loads(m)['type'] for m in messages if isinstance(m, str)], messages

    async def test_default_keeps_sentence_pause_in_one_turn(self):
        boundaries, messages = await self.send_frames([(True, 600), (False, 440), (True, 600), (False, 1000)], None)
        self.assertEqual(boundaries, ['utterance_start', 'utterance_end'])
        self.assertGreater(sum(len(m) for m in messages if isinstance(m, bytes)), 16000 * 2)

    async def test_longer_pause_separates_two_turns(self):
        boundaries, _ = await self.send_frames([(True, 600), (False, 1000), (True, 600), (False, 1000)])
        self.assertEqual(boundaries, ['utterance_start', 'utterance_end'] * 2)

    async def test_short_old_threshold_reproduces_sentence_split(self):
        boundaries, _ = await self.send_frames([(True, 600), (False, 440), (True, 600), (False, 1000)], '300')
        self.assertEqual(boundaries, ['utterance_start', 'utterance_end'] * 2)

    async def test_stop_flushes_open_turn_once(self):
        boundaries, _ = await self.send_frames([(True, 600), (False, 100)])
        self.assertEqual(boundaries, ['utterance_start', 'utterance_end'])

    async def test_silence_alone_creates_no_turn(self):
        boundaries, _ = await self.send_frames([(False, 1600)])
        self.assertEqual(boundaries, [])


if __name__ == '__main__':
    unittest.main()
