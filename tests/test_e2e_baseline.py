"""Meaningful regression checks for quality metrics, independent of model availability."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/e2e'))
from baseline import characterize, word_error


class BaselineMetricsTests(unittest.TestCase):
    def test_paris_output_cannot_hide_lost_microphone_question(self):
        events = [{'event': 'Baseline.Model', 'time': 1, 'message': {'content': 'Paris'},
                   'input': 'Paris', 'tools_offered': False},
                  {'event': 'E2E.AudioFetched', 'time': 2}]
        result = characterize({'id': 'capital', 'en': 'What is the capital of France?'}, 'en', 'mic', events)
        self.assertTrue(result['answer_check'])
        self.assertIn('expected_question_not_recognized', result['quality_flags'])

    def test_wrong_numeric_voice_is_flagged(self):
        events = [{'event': 'Baseline.Model', 'time': 1, 'message': {'content': '4'},
                   'input': 'deux plus deux', 'tools_offered': False},
                  {'event': 'ARIA.TTS.Language', 'time': 2, 'lang': 'en'},
                  {'event': 'E2E.AudioFetched', 'time': 3}]
        result = characterize({'id': 'arithmetic'}, 'fr', 'text', events)
        self.assertIn('wrong_tts_language', result['quality_flags'])

    def test_french_accents_punctuation_are_not_word_errors(self):
        self.assertEqual(word_error('Réponds : lumière !', 'reponds lumiere')['wer'], 0)

    def test_deletion_is_counted(self):
        self.assertEqual(word_error('turn on the light', 'turn the light')['errors'], 1)

    def test_token_limit_is_not_successful_answer_completion(self):
        events = [{'event': 'Baseline.Model', 'time': 1, 'message': {'content': 'Once upon'},
                   'input': 'Tell a story', 'done_reason': 'length', 'tools_offered': False},
                  {'event': 'E2E.AudioFetched', 'time': 2}]
        result = characterize({'id': 'story', 'en': 'Tell a story'}, 'en', 'wav', events)
        self.assertTrue(result['pipeline_completed'])
        self.assertTrue(result['truncated'])
        self.assertIsNone(result['answer_check'])

    def test_wrong_known_answer_is_a_quality_failure(self):
        events = [{'event': 'Baseline.Model', 'time': 1, 'message': {'content': '5'},
                   'input': 'two plus two', 'tools_offered': False}]
        result = characterize({'id': 'arithmetic'}, 'en', 'text', events)
        self.assertFalse(result['answer_check'])
        self.assertFalse(result['pipeline_completed'])
