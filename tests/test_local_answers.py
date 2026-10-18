from datetime import datetime, timezone

import pytest

from app.llm.local_answers import clock_answer, capability_answer
from types import SimpleNamespace


@pytest.mark.parametrize('instant, expected', [
    (datetime(2026, 1, 15, 23, 59, tzinfo=timezone.utc), '00:59'),
    (datetime(2026, 7, 15, 23, 59, tzinfo=timezone.utc), '01:59'),
])
@pytest.mark.parametrize('text, language', [
    ('Quelle heure est-il maintenant ? Réponds au format HH:MM, heure de Zurich.', 'fr'),
    ('What time is it now? Answer in HH:MM, Zurich time.', 'en'),
])
def test_clock_reads_instant_in_requested_zone(instant, expected, text, language):
    answer = clock_answer(text, now=instant)
    assert answer.text == expected
    assert answer.language == language


def test_clock_configuration_and_unavailable_zone(monkeypatch):
    instant = datetime(2026, 1, 15, 12, 4, tzinfo=timezone.utc)
    monkeypatch.setenv('ARIA_TIMEZONE', 'UTC')
    assert clock_answer('What time is it?', now=instant).text == 'It is 12:04.'
    monkeypatch.setenv('ARIA_TIMEZONE', 'Missing/Zone')
    assert clock_answer('Quelle heure est-il?', now=instant).kind == 'clock_unavailable'
    assert clock_answer('What time is it in Zurich?', now=instant).text == 'It is 13:04.'


@pytest.mark.parametrize('text', [
    'What time is it on Mars?', 'What time is it and turn on the light?',
    'Quelle heure est-il puis allume la lampe', 'What time is it tomorrow?',
    'What time is the train?', 'Explain what time is it means',
])
def test_clock_does_not_swallow_other_requests(text):
    assert clock_answer(text) is None


@pytest.mark.parametrize('text', ['Nomme les deux outils de fonction dont tu disposes.',
                                'Name the two function tools available to you.'])
def test_capabilities_follow_availability_without_invocation(text):
    started = SimpleNamespace(is_started=True)
    stopped = SimpleNamespace(is_started=False)
    assert 'find_devices' in capability_answer(text, started, stopped).text
    assert 'execute_command' not in capability_answer(text, started, stopped).text
    assert 'execute_command' in capability_answer(text, started, started).text
    assert 'find_devices' not in capability_answer(text, stopped, started).text
    assert 'find_devices' not in capability_answer(text, started, started, tools_enabled=False).text


def test_capabilities_log_only_and_mixed_intents():
    started = SimpleNamespace(is_started=True, execution_mode='log_only')
    reply = capability_answer('What can you do?', started, started).text
    assert 'only recorded' in reply and 'find_devices' not in reply
    assert capability_answer('What can you do and turn on the light?', started, started) is None
