"""Lexical aliases preserve identity; incomplete locations ask before any action."""
from dataclasses import replace

import pytest

from test_grounded_control import inventory
from app.llm.control import resolve, render
from app.plugins.inventory.selection import entity_text
from app.plugins.inventory.state_read import ground_read_source, select_read_commands, render_state
from test_shared_requests import server


@pytest.fixture
def seating(inventory):
    lamp = inventory.indexer.get_device('lamp')
    other = inventory.indexer.get_device('other')
    inventory.indexer.add_device(replace(lamp, name='Lampe Sofa', room='Salon'))
    inventory.indexer.add_device(replace(other, name='Lampe Fauteuil', room='Salon'))
    return inventory


@pytest.mark.parametrize('text,want', [
    ('Allume la lampe du divan.', 'lamp-on'),
    ('Turn off the armchair lamp.', 'other-off'),
    ('Turn on the couch lamp.', 'lamp-on'),
    ('Allume la lampe du fauteuil.', 'other-on'),
])
def test_seating_synonyms_preserve_distinct_targets(seating, text, want):
    plan = resolve(text, seating)
    assert plan.status == 'ready'
    assert plan.authorization.cmd_id == want


@pytest.mark.parametrize('text', [
    'Allume la lampe de la salle.', 'Allume la lampe de la pièce.',
    'Turn on the room light.', 'Allume la lampe dans la salle.',
])
def test_generic_room_requires_question_even_with_one_lamp(inventory, text):
    inventory.indexer.remove_device('other')
    lamp = inventory.indexer.get_device('lamp')
    inventory.indexer.add_device(replace(lamp, name='Lampe Salle à Manger', room='Cuisine'))
    plan = resolve(text, inventory)
    assert plan.status == 'ambiguous'
    assert plan.authorization is None
    assert render(plan).endswith('?')


def test_exact_generic_native_device_remains_addressable(inventory):
    lamp = inventory.indexer.get_device('lamp')
    inventory.indexer.add_device(replace(lamp, name='Lampe Salle'))
    assert resolve('Allume Lampe Salle.', inventory).authorization.cmd_id == 'lamp-on'


def test_declared_room_named_salle_remains_addressable(inventory):
    lamp = inventory.indexer.get_device('lamp')
    inventory.indexer.add_device(replace(lamp, name='Lampe Table', room='Salle'))
    assert resolve('Allume la lampe dans la pièce Salle.', inventory).authorization.cmd_id == 'lamp-on'


def test_read_of_declared_room_cannot_select_partial_name_elsewhere(inventory):
    for key, name, room in [('lamp', 'Lampe Table', 'Salle'), ('other', 'Lampe Salle à Manger', 'Bureau')]:
        d = inventory.indexer.get_device(key)
        state = replace(d.commands[0], cmd_id=key+'-state', type='info', subType='binary',
                        capability='light_state', execution=None, name='Etat')
        inventory.indexer.add_device(replace(d, name=name, room=room, commands=[*d.commands, state]))
    result = select_read_commands(inventory.indexer.get_snapshot(), 'state of the room Salle', 'state')
    assert result['status'] == 'ready'
    assert [c.cmd_id for c in result['commands']] == ['lamp-state']


def test_synonym_native_name_does_not_invent_room_constraint(inventory):
    lamp = inventory.indexer.get_device('lamp')
    other = inventory.indexer.get_device('other')
    inventory.indexer.add_device(replace(lamp, name='Lampe Salon', room='Bureau'))
    inventory.indexer.add_device(replace(other, name='Lampe Table', room='Salon'))
    assert resolve('Allume Lampe Séjour.', inventory).authorization.cmd_id == 'lamp-on'


def test_native_name_wins_over_synonym_collision(seating):
    lamp = seating.indexer.get_device('lamp')
    seating.indexer.add_device(replace(lamp, eq_id='native', name='Lampe Divan', commands=[
        replace(c, cmd_id='native-'+c.cmd_id, device_id='native') for c in lamp.commands]))
    assert resolve('Allume Lampe Divan.', seating).authorization.cmd_id == 'native-lamp-on'


def test_same_alias_on_two_devices_requires_question(seating):
    other = seating.indexer.get_device('other')
    seating.indexer.add_device(replace(other, name='Lampe Canapé'))
    plan = resolve('Allume la lampe du divan.', seating)
    assert plan.status == 'ambiguous'
    assert render(plan).endswith('?')


def test_synonym_cannot_erase_a_cross_domain_qualifier(seating):
    seating.indexer.remove_device('other')
    plan = resolve('Turn on the window blind lamp.', seating)
    assert plan.authorization is None


def test_cross_domain_name_modifier_matches_only_that_device(seating):
    other = seating.indexer.get_device('other')
    seating.indexer.add_device(replace(other, name='Lampe Store'))
    plan = resolve('Turn on the window blind lamp.', seating)
    assert plan.status == 'ready'
    assert plan.authorization.cmd_id == 'other-on'


@pytest.mark.parametrize('query', ['état de la salle', 'state of the room'])
def test_generic_read_requires_question(inventory, query):
    lamp = inventory.indexer.get_device('lamp')
    inventory.indexer.add_device(replace(lamp, name='Lampe Salle à Manger'))
    result = select_read_commands(inventory.indexer.get_snapshot(), query, 'state')
    assert result['status'] == 'ambiguous'
    assert result.get('commands', []) == []
    assert render_state(result, 'fr').endswith('?')
    assert 'pièce' in render_state(result, 'fr')


def test_model_cannot_expand_generic_room_into_specific_source(inventory):
    snapshot = inventory.indexer.get_snapshot()
    result = ground_read_source(snapshot, 'quelle temperature dans la salle', 'Lampe Salle à Manger', None)
    assert result['status'] == 'ambiguous'


@pytest.mark.parametrize('term', ['salle', 'pièce', 'manteau', 'pall', 'galerie', 'seat', 'chaise'])
def test_unreviewed_related_terms_are_not_synonym_aliases(term):
    assert entity_text(term) == term.replace('è', 'e')


@pytest.mark.parametrize('term,want', [
    ('divan', 'canape'), ('armchair', 'fauteuil'), ('chambre à coucher', 'chambre'),
    ('sitting room', 'salon'), ('coursive', 'couloir'), ('curtain', 'rideau'),
])
def test_reviewed_terms_are_normalized(term, want):
    assert entity_text(term) == want


@pytest.mark.asyncio
async def test_shared_request_asks_before_control(inventory, server):
    from types import SimpleNamespace
    from unittest.mock import Mock
    client, state = server
    lamp = inventory.indexer.get_device('lamp')
    inventory.indexer.add_device(replace(lamp, name='Lampe Salle à Manger'))
    actions = SimpleNamespace(execute_command=Mock())
    state.plugin_registry = SimpleNamespace(
        get_plugins_by_type=lambda _: [inventory],
        get_plugin=lambda _: actions)
    result = (await client.post('/api/llm/process', json={
        'text': 'Allume la lampe de la salle.', 'session_id': 'ambiguity',
        'use_tools': True, 'speak_response': False})).json()
    assert result['control_status'] == 'ambiguous'
    assert result['response'].endswith('?')
    actions.execute_command.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('text,language', [
    ('Quelle puissance indique la salle ?', 'fr'),
    ('What is the state of the room?', 'en'),
])
async def test_generic_read_question_does_not_depend_on_model(inventory, server, monkeypatch, text, language):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import app.main as runtime
    client, state = server
    inventory.read_state = AsyncMock()
    state.plugin_registry = SimpleNamespace(
        get_plugins_by_type=lambda _: [inventory], get_plugin=lambda _: None)
    model = AsyncMock(return_value={'content': 'This English response must not be used.'})
    monkeypatch.setattr(runtime, 'generate_llm_response', model)
    response = await client.post('/api/llm/process', json={
        'text': text, 'session_id': 'read-'+language,
        'use_tools': True, 'speak_response': True})
    result = response.json()
    assert result['success']
    assert result['response'] == render_state({'status': 'ambiguous'}, language)
    model.assert_not_awaited()
    inventory.read_state.assert_not_awaited()
    state.speech_output.speak.assert_awaited_once_with(result['response'], language=language)
