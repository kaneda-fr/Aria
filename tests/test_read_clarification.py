"""Clarifications must select inventory sources and resume real reads, never actions."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.plugins.inventory.provider import InventoryProvider
from app.plugins.inventory.state_read import ReadOnlyInventory
from app.llm.tools import execute_tool
from test_shared_requests import server


@pytest.fixture
def reader(tmp_path):
    def device(eq, name, domain, commands, room='Atelier'):
        return dict(source='jeedom_mqtt', eq_id=eq, name=name, room=room,
                    domain=domain, enabled=True, visible=True, seen_at=0, tags=[], commands=[
                        dict(cmd_id=cid, device_id=eq, name=label, type='info', subType=kind,
                             capability=capability, unit=unit, tags=[], visible=True)
                        for cid, label, kind, capability, unit in commands])
    devices = [
        device('1', 'Lampe Atelier', 'light', [
            ('101', 'Etat-1', 'binary', 'light_state', None),
            ('102', 'Etat-2', 'binary', 'light_state', None)]),
        device('2', 'Switch Atelier', 'generic', [
            ('201', 'Etat Binaire', 'binary', 'generic_info', None)]),
        device('3', 'Capteur Nord', 'weather', [
            ('301', 'Température', 'numeric', 'temperature', '°C'),
            ('302', 'Humidité', 'numeric', 'humidity', '%')], 'Garden'),
        device('4', 'Capteur Sud', 'weather', [
            ('401', 'Température', 'numeric', 'temperature', '°C'),
            ('402', 'Humidité', 'numeric', 'humidity', '%')], 'Garden'),
    ]
    path = tmp_path/'catalog.json'
    path.write_text(json.dumps({'devices': devices, 'last_update': 123}))
    result = ReadOnlyInventory('inventory', {'cache_path': str(path), 'sources': {}})
    InventoryProvider.start(result)
    result.apply_catalog({**result.get_snapshot(), 'connected': True, 'subscribed': True})
    reads = []
    def respond(request):
        if request.url.path == '/read':
            ids = json.loads(request.content)['cmd_ids']; reads.append(ids)
            values = {'101': 0.0, '102': 1.0, '201': 1.0, '301': 17.5, '302': 62.0, '401': 23.0, '402': 44.0}
            rows = []
            for cid in ids:
                cmd = result.indexer.get_command(cid); dev = result.indexer.get_device(cmd.device_id)
                rows.append(dict(cmd_id=cid, device_id=dev.eq_id, device_name=dev.name,
                                 name=cmd.name, unit=cmd.unit, status='ok', value=values[cid],
                                 observed_at=None, retrieved_at=1000))
            return httpx.Response(200, json={'success': True, 'status': 'read', 'readings': rows})
        if request.url.path == '/catalog':
            return httpx.Response(200, json={'unchanged': True, 'connected': True, 'subscribed': True,
                                            'catalog_version': 'fixture'})
        raise AssertionError(request.url)
    result.read_client = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url='http://local')
    return result, reads


@pytest.mark.asyncio
@pytest.mark.parametrize('utterance', [
    'Quel est l’état de la lumière dans Atelier ?',
    'What is the state of the light in Atelier?',
])
async def test_original_equipment_type_survives_model_query_omission(reader, utterance):
    provider, reads = reader
    result = await provider.read_state({'query': 'Atelier', 'metric': 'state', 'request_text': utterance})
    assert result['success'], result
    assert reads == [['101', '102']]


@pytest.mark.asyncio
@pytest.mark.parametrize('followup', [
    'La lampe.', 'Je parle de la lampe.', 'Lampe Atelier.',
    'The lamp, please.', 'I mean the lamp.', 'The first source, please.',
])
async def test_clarification_resumes_read_even_if_model_would_omit_tools(server, reader, monkeypatch, followup):
    import app.main as runtime
    from app.plugins.inventory.state_read import render_state
    from collections import deque
    c, state = server; provider, reads = reader
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    question = render_state({'status': 'ambiguous', 'devices': ['Lampe Atelier', 'Switch Atelier']}, 'fr')
    state.llm_history['choice'] = deque([
        {'role': 'user', 'content': '[speaker=person] Quel est l’état dans Atelier ?'},
        {'role': 'assistant', 'content': question}], maxlen=6)
    model = AsyncMock(return_value={'content': 'No device matches.', 'tool_calls': []})
    monkeypatch.setattr(runtime, 'generate_llm_response', model)
    response = (await c.post('/api/llm/process', json={
        'text': followup, 'session_id': 'choice', 'speak_response': False})).raise_for_status().json()
    assert response['success'] and 'Lampe Atelier' in response['response']
    assert reads == [['101', '102']]
    assert response['tool_calls_made'] == 1
    model.assert_not_awaited()
    assert state.llm_history['choice'][-1]['content'] == response['response']


@pytest.mark.asyncio
@pytest.mark.parametrize('followup', ['Capteur Sud.', 'Le second.', 'The second source, please.'])
async def test_source_choice_keeps_the_requested_metric_instead_of_reading_every_field(server, reader, monkeypatch, followup):
    import app.main as runtime
    from app.plugins.inventory.state_read import render_state
    from collections import deque
    c, state = server; provider, reads = reader
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    state.llm_history['metric-choice'] = deque([
        {'role': 'user', 'content': 'Quelle température indiquent les capteurs ?'},
        {'role': 'assistant', 'content': render_state({'status': 'ambiguous', 'devices': ['Capteur Nord', 'Capteur Sud']}, 'fr')}], maxlen=6)
    monkeypatch.setattr(runtime, 'generate_llm_response', AsyncMock(return_value={'content': 'Not found', 'tool_calls': []}))
    result = (await c.post('/api/llm/process', json={'text': followup, 'session_id': 'metric-choice', 'speak_response': False})).json()
    assert result['success'] and '23' in result['response']
    assert reads == [['401']]


@pytest.mark.asyncio
@pytest.mark.parametrize('utterance', ['Statut Lampe Atelier.', 'Status of Lampe Atelier.'])
async def test_explicit_state_label_reads_inventory_without_relying_on_model_tools(server, reader, monkeypatch, utterance):
    import app.main as runtime
    c, state = server; provider, reads = reader
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    model = AsyncMock(return_value={'content': 'Not found', 'tool_calls': []})
    monkeypatch.setattr(runtime, 'generate_llm_response', model)
    response = (await c.post('/api/llm/process', json={'text': utterance, 'session_id': 'state-label', 'speak_response': False})).json()
    assert response['success'] and 'Lampe Atelier' in response['response']
    assert reads == [['101', '102']]
    model.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['Quelle heure est-il ?', 'Why is the sky blue?', 'Turn off the lamp.', 'La lampe inconnue.'])
async def test_followup_does_not_guess_a_source_for_unrelated_or_action_requests(server, reader, monkeypatch, text):
    import app.main as runtime
    from app.plugins.inventory.state_read import render_state
    from collections import deque
    c, state = server; provider, reads = reader
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    state.llm_history['choice'] = deque([
        {'role': 'user', 'content': 'Read the state in Atelier.'},
        {'role': 'assistant', 'content': render_state({'status': 'ambiguous', 'devices': ['Lampe Atelier', 'Switch Atelier']})}], maxlen=6)
    monkeypatch.setattr(runtime, 'generate_llm_response', AsyncMock(return_value={'content': 'A general reply.', 'tool_calls': []}))
    await c.post('/api/llm/process', json={'text': text, 'session_id': 'choice', 'speak_response': False})
    assert not reads


@pytest.mark.asyncio
async def test_different_session_cannot_resume_another_clients_read(server, reader, monkeypatch):
    import app.main as runtime
    from collections import deque
    c, state = server; provider, reads = reader
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    state.llm_history['other-client'] = deque([
        {'role': 'user', 'content': 'Read the state in Atelier.'},
        {'role': 'assistant', 'content': 'Which source do you mean: Lampe Atelier, Switch Atelier ?'}], maxlen=6)
    monkeypatch.setattr(runtime, 'generate_llm_response', AsyncMock(return_value={'content': 'Which lamp?', 'tool_calls': []}))
    await c.post('/api/llm/process', json={'text': 'The first source.', 'session_id': 'new-client', 'speak_response': False})
    assert not reads


@pytest.mark.asyncio
@pytest.mark.parametrize('query,utterance', [
    ('lights at home', 'Liste des lumières à la maison.'),
    ('lights in the house', 'List the lights in the house.'),
    ('lampes à la maison', 'Montre les lampes à la maison.'),
])
async def test_household_scope_is_not_mistaken_for_a_device_name(reader, query, utterance):
    from dataclasses import replace
    provider, reads = reader
    lamp = provider.indexer.get_device('1')
    action = replace(lamp.commands[0], cmd_id='103', type='action', subType='other', capability='light_on')
    provider.indexer.add_device(replace(lamp, commands=[*lamp.commands, action]))
    result = await execute_tool('find_devices', {'query': query, 'request_text': utterance}, provider)
    assert result['success']
    assert {c['device_id'] for c in result['candidates']} == {'1'}
    assert not reads


@pytest.mark.asyncio
async def test_unknown_room_in_catalog_request_cannot_be_dropped_by_model(reader):
    from dataclasses import replace
    provider, reads = reader
    lamp = provider.indexer.get_device('1')
    action = replace(lamp.commands[0], cmd_id='103', type='action', subType='other', capability='light_on')
    provider.indexer.add_device(replace(lamp, commands=[*lamp.commands, action]))
    result = await execute_tool('find_devices', {'query': 'lights', 'request_text': 'List lights in Garage.'}, provider)
    assert not result['candidates']
    assert not reads


@pytest.mark.asyncio
async def test_clarification_keeps_the_channel_and_an_explicit_new_channel_can_override_it(server, reader, monkeypatch):
    import app.main as runtime
    from collections import deque
    from app.plugins.inventory.state_read import render_state
    c, state = server; provider, reads = reader
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    monkeypatch.setattr(runtime, 'generate_llm_response', AsyncMock(return_value={'content': 'Not found', 'tool_calls': []}))
    for followup, want in [('Lampe Atelier.', ['102']), ('Lampe Atelier canal 1.', ['101'])]:
        state.llm_history['channel-choice'] = deque([
            {'role': 'user', 'content': 'Quel est l’état dans Atelier canal 2 ?'},
            {'role': 'assistant', 'content': render_state({'status': 'ambiguous', 'devices': ['Lampe Atelier', 'Switch Atelier']}, 'fr')}], maxlen=6)
        response = (await c.post('/api/llm/process', json={'text': followup, 'session_id': 'channel-choice', 'speak_response': False})).json()
        assert response['success']
        assert reads[-1] == want


@pytest.mark.asyncio
async def test_removed_first_source_does_not_shift_the_ordinal_to_the_second(reader):
    from app.plugins.inventory.state_read import render_state
    provider, reads = reader
    question = render_state({'status': 'ambiguous', 'devices': ['Capteur Nord', 'Capteur Sud']})
    provider.indexer.remove_device('3')
    result = await provider.read_state({'query': 'Capteur Sud', 'metric': 'temperature',
        'request_text': 'The first source.', 'request_history': [
            {'role': 'user', 'content': 'Read the temperature.'},
            {'role': 'assistant', 'content': question}]})
    assert not result['success']
    assert not reads


@pytest.mark.asyncio
async def test_ordinary_model_question_is_not_a_trusted_read_clarification(server, reader, monkeypatch):
    import app.main as runtime
    from collections import deque
    c, state = server; provider, reads = reader
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    state.llm_history['not-a-choice'] = deque([
        {'role': 'user', 'content': 'Tell me a story.'},
        {'role': 'assistant', 'content': 'Should the story mention Lampe Atelier or Switch Atelier?'}], maxlen=6)
    model = AsyncMock(return_value={'content': 'A story about a lamp.', 'tool_calls': []})
    monkeypatch.setattr(runtime, 'generate_llm_response', model)
    response = (await c.post('/api/llm/process', json={'text': 'The lamp.', 'session_id': 'not-a-choice', 'speak_response': False})).json()
    assert response['response'] == 'A story about a lamp.'
    assert not reads
    model.assert_awaited_once()


@pytest.mark.asyncio
async def test_network_switch_can_be_chosen_by_its_actual_name_without_becoming_an_action(reader):
    from app.plugins.inventory.state_read import render_state
    provider, reads = reader
    result = await provider.read_state({'query': 'Switch Atelier', 'metric': 'state',
        'request_text': 'Switch Atelier', 'request_history': [
            {'role': 'user', 'content': 'What is the state in Atelier?'},
            {'role': 'assistant', 'content': render_state({'status': 'ambiguous', 'devices': ['Lampe Atelier', 'Switch Atelier']})}]})
    assert result['success'], result
    assert reads == [['201']]


@pytest.mark.asyncio
async def test_duplicate_source_names_never_authorize_an_ordinal_read(reader):
    from dataclasses import replace
    from app.plugins.inventory.state_read import render_state
    provider, reads = reader
    lamp = provider.indexer.get_device('1')
    provider.indexer.add_device(replace(lamp, eq_id='5', room='Cuisine', commands=[
        replace(lamp.commands[0], cmd_id='501', device_id='5')]))
    result = await provider.read_state({'query': 'Lampe Atelier', 'metric': 'state',
        'request_text': 'The first source.', 'request_history': [
            {'role': 'user', 'content': 'Read the state.'},
            {'role': 'assistant', 'content': render_state({'status': 'ambiguous', 'devices': ['Lampe Atelier', 'Lampe Atelier']})}]})
    assert not result['success'] and result['status'] == 'ambiguous'
    assert not reads


@pytest.mark.asyncio
async def test_network_switch_name_resumes_shared_read_without_a_model_action_guess(server, reader, monkeypatch):
    import app.main as runtime
    from collections import deque
    from app.plugins.inventory.state_read import render_state
    c, state = server; provider, reads = reader
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    state.llm_history['network-choice'] = deque([
        {'role': 'user', 'content': 'Read the state in Atelier.'},
        {'role': 'assistant', 'content': render_state({'status': 'ambiguous', 'devices': ['Lampe Atelier', 'Switch Atelier']})}], maxlen=6)
    model = AsyncMock(return_value={'content': 'A device changed.', 'tool_calls': []})
    monkeypatch.setattr(runtime, 'generate_llm_response', model)
    result = (await c.post('/api/llm/process', json={'text': 'Switch Atelier', 'session_id': 'network-choice', 'speak_response': False})).json()
    assert result['success'] and 'Switch Atelier' in result['response']
    assert reads == [['201']]
    model.assert_not_awaited()


@pytest.mark.parametrize('language,first,second', [('fr', 'canal 1', 'canal 2'), ('en', 'channel 1', 'channel 2')])
def test_multi_channel_read_reply_identifies_each_state_instead_of_two_unlabelled_values(language, first, second):
    from app.plugins.inventory.state_read import render_state
    result = {'status': 'read', 'readings': [
        {'device_id': '1', 'device_name': 'Lampe Atelier', 'name': 'Etat-1', 'metric': 'state', 'status': 'ok', 'value': 0.0},
        {'device_id': '1', 'device_name': 'Lampe Atelier', 'name': 'Etat-2', 'metric': 'state', 'status': 'ok', 'value': 1.0}]}
    reply = render_state(result, language)
    assert first in reply and second in reply


@pytest.mark.asyncio
@pytest.mark.parametrize('query,utterance', [
    ('Lampe Atelier', 'What options are available for Lampe Atelier?'),
    ('lights', 'What lights do you have around here?'),
    ('lights', 'List the lights that I can control at home.'),
])
async def test_catalog_keeps_model_interpretation_of_free_form_request_with_grounded_constraints(reader, query, utterance):
    from dataclasses import replace
    provider, reads = reader
    lamp = provider.indexer.get_device('1')
    action = replace(lamp.commands[0], cmd_id='103', type='action', subType='other', capability='light_on')
    provider.indexer.add_device(replace(lamp, commands=[*lamp.commands, action]))
    result = await execute_tool('find_devices', {'query': query, 'request_text': utterance}, provider)
    assert {r['device_id'] for r in result['candidates']} == {'1'}
    assert not reads


@pytest.mark.asyncio
@pytest.mark.parametrize('utterance', ['Liste des lumières dans Atelier.', 'List the lights in Atelier.'])
async def test_catalog_preserves_user_device_type_when_model_omits_it(reader, utterance):
    from dataclasses import replace
    provider, reads = reader
    for device_id, cap, action_id in [('1', 'light_on', '103'), ('2', 'switch_on', '203')]:
        device = provider.indexer.get_device(device_id)
        action = replace(device.commands[0], cmd_id=action_id, type='action',
                         subType='other', capability=cap)
        provider.indexer.add_device(replace(device, commands=[*device.commands, action]))
    result = await execute_tool('find_devices', {'query': 'Atelier', 'request_text': utterance}, provider)
    assert {r['device_id'] for r in result['candidates']} == {'1'}
    assert not reads


@pytest.mark.asyncio
@pytest.mark.parametrize('utterance', [
    'La lampe Lampe Atelier dans Atelier est-elle actuellement allumée ?',
    'Is Lampe Atelier in Atelier switched on right now?',
])
async def test_room_before_state_predicate_is_not_extended_into_a_nonexistent_room(reader, utterance):
    provider, reads = reader
    result = await execute_tool('get_device_state', {
        'query': 'Lampe Atelier', 'room': 'Atelier', 'metric': 'state',
        'request_text': utterance}, provider)
    assert result['status'] == 'read'
    assert reads == [['101', '102']]


@pytest.mark.asyncio
@pytest.mark.parametrize('suffix', ['Annexe', 'Est', 'Garage allumée', 'East switched on'])
async def test_known_room_prefix_does_not_authorize_an_unknown_room_qualifier(reader, suffix):
    provider, reads = reader
    result = await execute_tool('get_device_state', {
        'query': 'Lampe Atelier', 'room': 'Atelier', 'metric': 'state',
        'request_text': 'État de Lampe Atelier dans la pièce Atelier ' + suffix}, provider)
    assert result['status'] == 'not_found'
    assert not reads


@pytest.mark.asyncio
@pytest.mark.parametrize('name,utterance', [
    ('Maison', 'Liste des lumières à la maison.'),
    ('Home', 'List lights at home.'),
])
async def test_household_catalog_scope_is_not_a_same_named_inventory_device(reader, name, utterance):
    from dataclasses import replace
    provider, reads = reader
    lamp = provider.indexer.get_device('1')
    action = replace(lamp.commands[0], cmd_id='103', type='action', subType='other', capability='light_on')
    provider.indexer.add_device(replace(lamp, commands=[*lamp.commands, action]))
    for eq in ['5', '6']:
        device = provider.indexer.get_device('2')
        command = replace(device.commands[0], cmd_id=eq+'01', device_id=eq)
        provider.indexer.add_device(replace(device, eq_id=eq, name=name, commands=[command]))
    result = await execute_tool('find_devices', {'query': 'lights', 'request_text': utterance}, provider)
    assert {r['device_id'] for r in result['candidates']} == {'1'}
    assert not reads


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['Lamp at home', 'Lampe à la maison'])
async def test_catalog_preserves_a_full_native_name_containing_household_scope(reader, name):
    from dataclasses import replace
    provider, reads = reader
    lamp = provider.indexer.get_device('1')
    action = replace(lamp.commands[0], cmd_id='103', type='action', subType='other', capability='light_on')
    provider.indexer.add_device(replace(lamp, name=name, commands=[*lamp.commands, action]))
    result = await execute_tool('find_devices', {
        'query': name, 'request_text': 'List controls for ' + name}, provider)
    assert {r['device_name'] for r in result['candidates']} == {name}
    assert not reads


@pytest.mark.asyncio
@pytest.mark.parametrize('text', [
    'State a fact about planets.',
    'State the result of two plus two.',
    'État de la matière : le glaçon est-il solide ?',
    'State light as a wave phenomenon.',
])
async def test_state_word_in_general_questions_does_not_force_a_device_read(server, reader, monkeypatch, text):
    import app.main as runtime
    c, state = server
    provider, reads = reader
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    model = AsyncMock(return_value={'content': 'A general reply.', 'tool_calls': []})
    monkeypatch.setattr(runtime, 'generate_llm_response', model)
    response = await c.post('/api/llm/process', json={
        'text': text, 'session_id': 'general-state-word', 'speak_response': False})
    assert response.json()['response'] == 'A general reply.'
    model.assert_awaited_once()
    assert not reads


@pytest.mark.asyncio
async def test_explicit_channel_does_not_hide_a_known_state_target_from_shared_dispatch(server, reader, monkeypatch):
    import app.main as runtime
    c, state = server
    provider, reads = reader
    state.plugin_registry = SimpleNamespace(get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    model = AsyncMock(return_value={'content': 'Which device?', 'tool_calls': []})
    monkeypatch.setattr(runtime, 'generate_llm_response', model)
    await c.post('/api/llm/process', json={
        'text': 'Status of Lampe Atelier channel 2.', 'session_id': 'known-channel', 'speak_response': False})
    assert reads == [['102']]
    model.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('utterance', [
    'État de la lumière Switch Atelier.',
    'State of the light Switch Atelier.',
])
@pytest.mark.parametrize('clarification', [False, True])
async def test_explicit_type_outside_native_name_cannot_select_a_different_equipment_type(reader, utterance, clarification):
    provider, reads = reader
    history = [
        {'role': 'user', 'content': 'State Atelier.'},
        {'role': 'assistant', 'content': 'Which source do you mean: Lampe Atelier, Switch Atelier ?'},
    ] if clarification else []
    result = await execute_tool('get_device_state', {
        'query': 'Switch Atelier', 'metric': 'state', 'request_text': utterance,
        'request_history': history}, provider)
    assert result['status'] == 'not_found'
    assert not reads

@pytest.mark.asyncio
async def test_type_word_inside_native_name_does_not_override_native_identity(reader):
    from dataclasses import replace
    provider, reads = reader
    device = provider.indexer.get_device('2')
    provider.indexer.add_device(replace(device, name='Lampe Technique'))
    result = await execute_tool('get_device_state', {
        'query': 'Lampe Technique', 'metric': 'state', 'request_text': 'État de Lampe Technique.'}, provider)
    assert result['status'] == 'read'
    assert reads == [['201']]
