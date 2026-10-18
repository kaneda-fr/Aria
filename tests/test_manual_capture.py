"""Controlled capture lifecycle without opening audio hardware."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import wave

import numpy as np
from scripts.e2e import manual_capture as capture_tool


class ManualCaptureTests(unittest.TestCase):
    def test_ready_follows_audio_and_stop_closes_stream(self):
        events = []
        class Stream:
            def __init__(self, **kwargs):
                self.callback = kwargs['callback']
                self.rate = kwargs['samplerate']
            def __enter__(self):
                events.append('audio')
                self.callback(np.full((960, 1), .125, dtype=np.float32), 960, None, '')
                return self
            def __exit__(self, *args):
                events.append('closed')
        def announce(message, **kwargs):
            if message.startswith('READY'): events.append('ready')
            if message.startswith('STOPPED'): events.append('stopped')
        with patch.object(capture_tool, '_select_input_device', return_value=0), \
             patch.object(capture_tool.sd, 'query_devices', return_value={'name':'fake', 'default_samplerate':96000}), \
             patch.object(capture_tool.sd, 'InputStream', Stream):
            samples, rate, metadata = capture_tool.capture('fake', 3, lambda: True, announce)
        self.assertEqual(events, ['audio', 'ready', 'closed', 'stopped'])
        self.assertEqual(rate, 96000)
        self.assertEqual(metadata['frames'], 960)
        np.testing.assert_array_equal(samples, np.full(960, .125, dtype=np.float32))

    def test_native_samples_and_analysis_audio_saved_with_hashes(self):
        samples = np.sin(np.arange(9600) * .05).astype(np.float32) * .2
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            hashes = capture_tool.save_audio(root, samples, 96000)
            np.testing.assert_array_equal(np.load(root / 'native-float32.npy'), samples)
            for filename, rate, frames in [('native.wav', 96000, 9600), ('analysis-16k.wav', 16000, 1600)]:
                with wave.open(str(root / filename)) as audio:
                    self.assertEqual((audio.getframerate(), audio.getnframes(), audio.getnchannels()), (rate, frames, 1))
                self.assertEqual(len(hashes[filename]), 64)


if __name__ == '__main__':
    unittest.main()
