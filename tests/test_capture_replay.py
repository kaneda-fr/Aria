"""Ground-truth and byte-trace regressions; no capture, server, or model needed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
import wave

import numpy as np
from scripts.e2e.capture_replay import load_capture, capture_checks, WireCapture


class CaptureReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        np.save(self.root/'native-float32.npy', np.zeros(1600, dtype=np.float32))
        self.digest = hashlib.sha256((self.root/'native-float32.npy').read_bytes()).hexdigest()
        self.manifest = {'expected_text': 'What is the capital of France?',
                         'listener_notes': 'clean', 'native_rate': 16000, 'frames': 1600,
                         'sha256': {'native-float32.npy': self.digest}}
        (self.root/'capture.json').write_text(json.dumps(self.manifest))

    def test_displayed_prompt_and_clean_notes_are_not_confirmed_reference(self):
        with self.assertRaisesRegex(ValueError, 'user-confirmed'):
            load_capture(self.root)

    def test_actual_spoken_reference_overrides_displayed_prompt(self):
        (self.root/'reference.json').write_text(json.dumps({
            'confirmed_by':'user', 'spoken_text':'The capital of France is Paris.',
            'native_sha256':self.digest}))
        result = load_capture(self.root)
        self.assertEqual(result['reference'], 'The capital of France is Paris.')
        self.assertEqual(result['manifest']['expected_text'], self.manifest['expected_text'])

    def test_reference_cannot_be_reused_for_different_audio(self):
        (self.root/'reference.json').write_text(json.dumps({
            'confirmed_by':'user', 'spoken_text':'Bonjour', 'native_sha256':'wrong'}))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            load_capture(self.root)

    def test_extra_answer_in_merged_transcript_is_flagged(self):
        text = 'Quelle est la capitale de la France? La capitale de la France est Paris.'
        result = capture_checks('Quelle est la capitale de la France?', [{'pcm_sha256':'abc'}],
            [{'event':'Baseline.ASR', 'pcm_sha256':'abc', 'text':text}], [text])
        self.assertEqual(result['input_trace_flags'], ['confirmed_input_mismatch'])

    def test_clean_trace_accepts_punctuation_and_detects_changed_stage(self):
        events = [{'event':'Baseline.ASR','pcm_sha256':'abc','text':'Bonjour!'}]
        result = capture_checks('bonjour', [{'pcm_sha256':'abc'}], events, ['Bonjour!'])
        self.assertTrue(result['confirmed_input_matches'])
        result = capture_checks('bonjour', [{'pcm_sha256':'different'}], events, ['Paris'])
        self.assertEqual(set(result['input_trace_flags']), {
            'asr_audio_hash_mismatch','asr_to_request_mismatch','confirmed_input_mismatch'})


class WireCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_records_exact_sent_bytes_and_does_not_duplicate_stop(self):
        with tempfile.TemporaryDirectory() as name:
            wire = WireCapture(SimpleNamespace(send=AsyncMock()), Path(name))
            await wire.send(json.dumps({'type':'utterance_start'}))
            pcm = b'\x01\x00' * 160
            await wire.send(pcm)
            await wire.send(json.dumps({'type':'utterance_end'}))
            await wire.send(json.dumps({'type':'stop'}))
            self.assertEqual(len(wire.segments), 1)
            self.assertEqual(wire.segments[0]['pcm_sha256'], hashlib.sha256(pcm).hexdigest())
            with wave.open(wire.segments[0]['path']) as audio:
                self.assertEqual(audio.readframes(audio.getnframes()), pcm)


if __name__ == '__main__':
    unittest.main()
