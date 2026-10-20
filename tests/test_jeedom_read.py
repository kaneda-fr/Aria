"""Read path must never turn a lookup into a physical command."""
import importlib.util
import json
from types import SimpleNamespace

import pytest


def modules():
    assert importlib.util.find_spec('app.plugins.inventory.read_bridge'), 'read-only bridge is missing'
    from app.plugins.inventory.read_bridge import JeedomReadBridge
    from app.plugins.inventory.state_read import select_read_commands, render_state
    return JeedomReadBridge, select_read_commands, render_state


@pytest.fixture
def cache(tmp_path):
    def device(eq, name, backend, enabled=True):
        return dict(source='jeedom_mqtt', eq_id=eq, name=name, room='Garden',
                    domain='default', enabled=enabled, visible=True, seen_at=0,
                    tags=['backend:' + backend], commands=[
                        dict(cmd_id=eq+'1', device_id=eq, name='Température', type='info',
                             subType='numeric', capability='temperature', unit='°C', tags=[]),
                        dict(cmd_id=eq+'2', device_id=eq, name='Refresh', type='action',
                             subType='other', capability=None, unit=None, tags=[]),
                        dict(cmd_id=eq+'3', device_id=eq, name='Température +1 jour', type='info',
                             subType='numeric', capability=None, unit='°C', tags=[]),
                    ])
    p = tmp_path/'catalog.json'
    p.write_text(json.dumps({'devices': [device('1', 'Meteo North', 'weather'),
                                        device('2', 'Meteo South', 'MeteoFull'),
                                        device('3', 'Meteo Disabled', 'weather', False)],
                             'last_update': 123}))
    return p


def bridge(cache):
    cls, _, _ = modules()
    b = cls(cache, host='127.0.0.1')
    b.connected = True
    b.subscribed = True
    class MQTT:
        def __init__(self): self.sent = []
        def publish(self, topic, payload, qos, retain):
            self.sent.append((topic, payload, qos, retain))
            b.receive(topic.replace('/get/', '/value/'), b'12.5', retained=False)
            return SimpleNamespace(rc=0)
    b.client = MQTT()
    return b


def test_read_get_only_and_measurement_time_is_unknown(cache):
    b = bridge(cache)
    result = b.read(['11'], timeout=.05)
    assert result['success']
    assert result['readings'][0]['value'] == 12.5
    assert result['readings'][0]['unit'] == '°C'
    assert result['readings'][0]['observed_at'] is None
    assert result['readings'][0]['retrieved_at'] > 0
    assert b.client.sent == [('jeedom/cmd/get/11', '', 1, False)]


@pytest.mark.parametrize('ids', [['12'], ['31'], ['999'], ['11/../../set/12'], ['11']*9])
def test_rejects_actions_disabled_unknown_and_oversized_reads(cache, ids):
    b = bridge(cache)
    assert not b.read(ids, timeout=.01)['success']
    assert b.client.sent == []


def test_retained_unsolicited_value_cannot_satisfy_a_new_read(cache):
    b = bridge(cache)
    b.client.publish = lambda *a, **k: SimpleNamespace(rc=0)
    b.receive('jeedom/cmd/value/11', b'99', retained=True)
    result = b.read(['11'], timeout=.01)
    assert result['readings'][0]['status'] == 'unavailable'
    assert 'value' not in result['readings'][0]


def test_disconnect_and_malformed_value_are_not_silent_success(cache):
    b = bridge(cache)
    b.connected = False
    assert not b.read(['11'], timeout=.01)['success']
    b.connected = True
    b.client.publish = lambda *a, **k: (b.receive('jeedom/cmd/value/11', b'NaN', False)
                                       or SimpleNamespace(rc=0))
    assert b.read(['11'], timeout=.01)['readings'][0]['status'] == 'unavailable'


def test_live_discovery_updates_catalog_and_removes_old_commands(cache):
    b = bridge(cache)
    raw = {'id': '1', 'name': 'Meteo Renamed', 'isEnable': '1', 'isVisible': '1',
           'object_name': 'Garden', 'eqType_name': 'weather', 'configuration': {},
           'cmds': {'19': {'id': '19', 'eqLogic_id': '1', 'name': 'Humidity',
                           'type': 'info', 'subType': 'numeric', 'unite': '%'}}}
    b.receive('jeedom/discovery/eqLogic/1', json.dumps(raw).encode(), False)
    snap = b.catalog()
    assert snap['devices']['1']['name'] == 'Meteo Renamed'
    assert '11' not in snap['commands']
    assert '19' in snap['commands']
    assert snap['live_discovery_count'] == 1
    assert snap['catalog_complete'] is False


def test_disabled_discovery_removes_a_previously_enabled_device(cache):
    b = bridge(cache)
    b.receive('jeedom/discovery/eqLogic/1', b'{"id":"1","isEnable":"0"}', False)
    assert '1' not in b.catalog()['devices']
    assert not b.read(['11'], timeout=.01)['success']


def test_discovery_catalog_survives_restart_without_persisting_sensor_values(cache):
    b = bridge(cache)
    b.receive('jeedom/discovery/eqLogic/1', b'{"id":"1","isEnable":"0"}', False)
    assert callable(getattr(b, 'save_catalog', None)), 'live catalog persistence is missing'
    b.save_catalog()
    cls, _, _ = modules()
    restored = cls(cache, host='127.0.0.1')
    assert '1' not in restored.catalog()['devices']
    assert restored.catalog()['states'] == {}
    assert restored.seed_last_update > 123


@pytest.mark.asyncio
async def test_bridge_http_contract_has_no_action_endpoint(cache):
    import httpx
    b = bridge(cache)
    from app.plugins.inventory.read_bridge import create_app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(b)),
                                 base_url='http://local') as client:
        assert (await client.post('/read', json={'cmd_ids': ['11']})).json()['readings'][0]['value'] == 12.5
        assert not (await client.post('/read', json={'cmd_ids': ['12']})).json()['success']
        assert (await client.post('/action', json={'cmd_id': '12'})).status_code == 404
        assert (await client.post('/read', content='x'*1025)).status_code == 413
        assert (await client.get('/catalog')).json()['catalog_complete'] is False


@pytest.mark.asyncio
async def test_unchanged_catalog_skips_definitions_but_reports_health_and_updates(cache):
    import httpx
    b = bridge(cache)
    from app.plugins.inventory.read_bridge import create_app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(b)),
                                 base_url='http://local') as client:
        first = (await client.get('/catalog')).json()
        assert 'catalog_version' in first, 'conditional catalog retrieval is missing'
        version = first['catalog_version']
        same = (await client.get('/catalog', params={'version': version})).json()
        assert same['unchanged'] and 'devices' not in same
        b.connected = False
        assert not (await client.get('/catalog', params={'version': version})).json()['connected']
        b.receive('jeedom/discovery/eqLogic/1', b'{"id":"1","isEnable":"0"}', False)
        changed = (await client.get('/catalog', params={'version': version})).json()
        assert 'devices' in changed and '1' not in changed['devices']


@pytest.mark.asyncio
async def test_inventory_adapter_reuses_its_read_connection_and_closes_it(cache):
    import httpx
    b = bridge(cache)
    from app.plugins.inventory.read_bridge import create_app
    from app.plugins.inventory.state_read import ReadOnlyInventory
    from app.plugins.inventory.provider import InventoryProvider
    reader = ReadOnlyInventory('inventory', {'cache_path': str(cache), 'sources': {}})
    InventoryProvider.start(reader)
    reader.read_socket = '/no-new-connection-allowed.sock'
    reader.apply_catalog(b.catalog())
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(b)),
                               base_url='http://local')
    reader.read_client = client
    try:
        result = await reader.read_state({'query': 'Meteo North', 'metric': 'temperature'})
        assert result['success'] and result['readings'][0]['value'] == 12.5
        assert result['readings'][0]['metric'] == 'temperature'
        await reader.close_read_connection()
        assert client.is_closed
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_explicit_requested_measurement_overrides_wrong_model_metric(cache):
    import httpx
    b = bridge(cache)
    from app.plugins.inventory.read_bridge import create_app
    from app.plugins.inventory.state_read import ReadOnlyInventory
    from app.plugins.inventory.provider import InventoryProvider
    reader = ReadOnlyInventory('inventory', {'cache_path': str(cache), 'sources': {}})
    InventoryProvider.start(reader)
    reader.apply_catalog(b.catalog())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(b)),
                                 base_url='http://local') as client:
        reader.read_client = client
        result = await reader.read_state({'query': 'Meteo North', 'metric': 'condition',
                                         'request_text': 'What temperature does Meteo North report?'})
    assert result['success']
    assert [r['cmd_id'] for r in result['readings']] == ['11']


def test_condition_text_is_not_confused_with_numeric_condition_code(cache):
    from dataclasses import replace
    b = bridge(cache)
    _, select, _ = modules()
    device = b.provider.indexer.get_device('1')
    text = replace(device.commands[0], cmd_id='14', name='Condition', subType='string', capability=None, unit=None)
    code = replace(text, cmd_id='15', name='Numéro condition', subType='numeric')
    b.provider.indexer.add_device(replace(device, commands=[text, code]))
    result = select(b.provider.indexer.get_snapshot(), 'Meteo North', metric='condition')
    assert [c.cmd_id for c in result['commands']] == ['14']


@pytest.mark.parametrize('query', ['Meteo North', 'weather North', 'météo North'])
def test_named_weather_selects_current_temperature_not_refresh_or_forecast(cache, query):
    b = bridge(cache)
    _, select, _ = modules()
    result = select(b.provider.indexer.get_snapshot(), query, metric='temperature')
    assert result['status'] == 'ready'
    assert [c.cmd_id for c in result['commands']] == ['11']


def test_weather_requires_source_clarification_and_never_invents_forecast(cache):
    b = bridge(cache)
    _, select, _ = modules()
    assert select(b.provider.indexer.get_snapshot(), 'weather')['status'] == 'ambiguous'
    assert select(b.provider.indexer.get_snapshot(), 'Meteo North demain')['status'] == 'unsupported_period'
    assert select(b.provider.indexer.get_snapshot(), 'Meteo East')['status'] == 'not_found'


def test_weather_source_followup_can_use_just_its_location(cache):
    b = bridge(cache)
    _, select, _ = modules()
    result = select(b.provider.indexer.get_snapshot(), 'North')
    assert result['status'] == 'ready'
    assert [c.cmd_id for c in result['commands']] == ['11']


def test_metric_filters_location_matches_before_declaring_ambiguity(cache):
    from dataclasses import replace
    b = bridge(cache)
    _, select, _ = modules()
    original = b.provider.indexer.get_device('1')
    clock = replace(original, eq_id='9', name='Clock North', tags=set(),
                    commands=[replace(original.commands[0], cmd_id='91', device_id='9',
                                      name='Time', capability=None, subType='string', unit=None)])
    b.provider.indexer.add_device(clock)
    result = select(b.provider.indexer.get_snapshot(), 'North', metric='temperature')
    assert result['status'] == 'ready'
    assert [c.cmd_id for c in result['commands']] == ['11']


@pytest.mark.parametrize('language,value,expected,absent', [
    ('en', 'Pluie faible', 'light rain', 'Pluie faible'),
    ('fr', 'Light rain', 'pluie faible', 'Light rain'),
])
def test_weather_condition_enum_is_spoken_in_request_language(language, value, expected, absent):
    _, _, render = modules()
    text = render({'success': True, 'status': 'read', 'readings': [{
        'status': 'ok', 'metric': 'condition', 'value': value, 'device_name': 'Weather North',
        'observed_at': None}]}, language)
    assert expected in text.lower()
    assert absent not in text


def test_unrecognized_weather_text_is_not_guessed_or_spoken_in_another_language():
    _, _, render = modules()
    text = render({'success': True, 'status': 'read', 'readings': [{
        'status': 'ok', 'metric': 'condition', 'value': 'lluvia débil',
        'device_name': 'Weather North', 'observed_at': None}]}, 'en')
    assert 'lluvia' not in text and 'condition' in text.lower()


@pytest.mark.parametrize('language,value,expected', [
    ('en', 'Pluies éparses à proximité', 'patchy rain nearby'),
    ('fr', 'Patchy rain nearby', 'pluies éparses à proximité'),
])
def test_offline_provider_condition_vocabulary_covers_localized_labels(language, value, expected):
    _, _, render = modules()
    text = render({'success': True, 'status': 'read', 'readings': [{
        'status': 'ok', 'metric': 'condition', 'value': value,
        'device_name': 'Weather North', 'observed_at': None}]}, language)
    assert expected in text.lower()


@pytest.mark.asyncio
async def test_original_request_period_cannot_be_dropped_by_tool_arguments(cache):
    modules()
    from app.plugins.inventory.state_read import ReadOnlyInventory
    from app.plugins.inventory.provider import InventoryProvider
    reader = ReadOnlyInventory('inventory', {'cache_path': str(cache), 'sources': {}})
    InventoryProvider.start(reader)
    async def refresh(**kwargs): pass
    reader.refresh_live_inventory = refresh
    result = await reader.read_state({'query': 'Meteo North', 'metric': 'temperature',
                                     'request_text': 'What is the weather tomorrow in North?'})
    assert result['status'] == 'unsupported_period'


@pytest.mark.parametrize('language, fragment', [('fr', 'mesure'), ('en', 'measurement')])
def test_reply_uses_actual_value_and_discloses_unknown_measurement_time(cache, language, fragment):
    b = bridge(cache)
    _, _, render = modules()
    result = b.read(['11'], timeout=.05)
    text = render(result, language)
    assert '12.5' in text and '°C' in text and 'Meteo North' in text
    assert fragment in text.lower()
    assert 'Refresh' not in text


@pytest.mark.asyncio
async def test_tool_is_available_only_with_a_read_provider_and_dispatches(cache):
    modules()
    from app.llm.tools import available_tools, execute_tool
    plain = SimpleNamespace(is_started=True)
    assert 'get_device_state' not in [t['function']['name'] for t in available_tools(plain)]
    class Reader:
        is_started = True
        async def read_state(self, args): return {'success': True, 'readings': [{'value': 7}]}
    reader = Reader()
    assert 'get_device_state' in [t['function']['name'] for t in available_tools(reader)]
    result = await execute_tool('get_device_state', {'query': 'weather'}, reader)
    assert result['readings'][0]['value'] == 7
