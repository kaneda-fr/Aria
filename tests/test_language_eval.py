"""Verify experiment accounting and shared neutral handling without optional models."""
import unittest

from scripts.e2e.language_eval import neutral_or_short, score, segments


class LanguageEvaluationOracleTests(unittest.TestCase):
    def test_separates_failure_types(self):
        cases = [
            dict(id='voice', category='short', expected='fr', hint='fr', text='Bonjour'),
            dict(id='refusal', category='short', expected='en', hint='en', text='Thank you'),
            dict(id='miss', category='mixed', expected='mixed', hint='en', text='Hello et merci'),
            dict(id='caught', category='foreign', expected='unsupported', hint='fr', text='Hola'),
            dict(id='neutral', category='neutral', expected='neutral', hint='fr', text='42'),
        ]
        result = score(cases, ['en', 'unknown', 'en', 'es', 'fr'])
        self.assertEqual(result['correct'], 2)
        self.assertEqual(result['false_fallback'], 1)
        self.assertEqual(result['wrong_voice'], 1)
        self.assertEqual(result['missed_bad'], 1)
        self.assertEqual(result['neutral_errors'], 0)
        self.assertEqual(len(result['errors']), 3)

    def test_non_latin_sentence_is_not_a_neutral_name(self):
        self.assertEqual(neutral_or_short('这是一个完整的句子。', 'en', set()), 'unsupported')

    def test_identifier_preserves_request_voice(self):
        for hint in ('fr', 'en'):
            self.assertEqual(neutral_or_short('X-503', hint, set()), hint)

    def test_inventory_name_preserves_request_voice(self):
        self.assertEqual(neutral_or_short('Garden room.', 'fr', {'garden room'}), 'fr')

    def test_short_confirmation_has_language(self):
        self.assertEqual(neutral_or_short('Merci !', 'en', set()), 'fr')

    def test_window_work_is_bounded(self):
        text = ' '.join(f'word{x}' for x in range(1000))
        self.assertLessEqual(len(segments(text, True)), 12)


if __name__ == '__main__':
    unittest.main()
