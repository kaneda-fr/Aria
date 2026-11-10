"""Held-out grammatical variants must keep inventory scope and read intent."""
import pytest

from test_light_questions import lighting
from app.plugins.inventory.light_queries import resolve_light_query


@pytest.mark.parametrize('text,wanted', [
    ('Are there any lights still on?', 'on'),
    ('Are there any lamps off?', 'off'),
    ('Is there a light on anywhere?', 'on'),
    ('Quel est le statut de mon éclairage ?', None),
    ('Mon éclairage est-il allumé ?', 'on'),
    ('Est-ce que mon éclairage est éteint ?', 'off'),
])
def test_existential_and_collective_lighting_scope(lighting, text, wanted):
    provider, calls, _ = lighting
    query = resolve_light_query(text, provider.indexer.get_snapshot())
    assert query is not None and query.status == 'ready'
    assert {eq for eq, _, _ in query.targets} == {'1', '3', '4', '7', '8'}
    assert query.wanted == wanted
    assert not calls

@pytest.mark.parametrize('text', [
    'Combien font douze fois sept ?',
    'What is twelve times seven?',
    'And what is twelve times seven?',
    'Combien de continents y a-t-il ?',
    'How many days are in a week?',
    'What is the capital of France?',
    'What time is it?',
    'Tell me about the moon.',
    'In French, please.',
    'In what year did the moon landing happen?',
    'Dans quel pays se trouve Paris ?',
    'In a few words explain gravity.',
])
def test_unrelated_question_never_inherits_a_lighting_scope(lighting, text):
    provider, calls, _ = lighting
    history = [{'role': 'user', 'content': 'List my lights.'},
               {'role': 'assistant', 'content': 'Lights: 5 devices available.'}]
    assert resolve_light_query(text, provider.indexer.get_snapshot(), history) is None
    assert not calls

@pytest.mark.parametrize('text', [
    'Now turn that same light off.',
    'Maintenant éteins cette même lumière.',
    'Could you now switch that light off?',
    'Please turn the office light on.',
    'Turn all the lights off.',
    'Do not turn off the lights.',
    'Ne pas allumer les lumières.',
    'Now dim the office light to 35 percent.',
])
def test_supported_imperatives_belong_to_control_not_state(lighting, text):
    from app.llm.control import resolve as resolve_control
    assert resolve_control(text, None) is not None
    provider, calls, _ = lighting
    history = [{'role': 'user', 'content': 'List my lights.'},
               {'role': 'assistant', 'content': 'Lights: 5 devices available.'}]
    assert resolve_light_query(text, provider.indexer.get_snapshot(), history) is None
    assert not calls
