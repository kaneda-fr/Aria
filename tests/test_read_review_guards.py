"""Cross-speaker clarification and delayed bridge responses fail closed."""
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from test_jeedom_read import bridge, cache
from test_read_clarification import reader
from test_shared_requests import server


@pytest.mark.asyncio
@pytest.mark.parametrize('current,prior,expected', [
    ('alice', ('alice',), [['101', '102']]),
    ('bob', ('alice',), []),
    ('alice', ('alice', 'bob'), []),
])
async def test_source_choice_history_stops_at_other_speaker(
        server, reader, monkeypatch, current, prior, expected):
    import app.main as runtime
    from app.plugins.inventory.state_read import render_state
    client, state = server
    provider, reads = reader
    monkeypatch.setenv('ARIA_LLM_HISTORY_MAX_MESSAGES', '6')
    state.plugin_registry = SimpleNamespace(
        get_plugins_by_type=lambda _: [provider], get_plugin=lambda _: None)
    question = render_state({'status': 'ambiguous',
                             'devices': ['Lampe Atelier', 'Switch Atelier']}, 'en')
    history = []
    for speaker in prior:
        history.extend([{'role': 'user', 'content': f'[speaker={speaker}] What is the state in Atelier?'},
                        {'role': 'assistant', 'content': question}])
    state.llm_history['shared'] = deque(history, maxlen=6)
    model = AsyncMock(return_value={'content': '', 'tool_calls': [
        {'function': {'name': 'get_device_state', 'arguments': {'query': 'the first source'}}}]})
    monkeypatch.setattr(runtime, 'generate_llm_response', model)
    try:
        result = (await client.post('/api/llm/process', json={
            'text': 'The first source.', 'session_id': 'shared',
            'speaker_user': current, 'speak_response': False,
        })).raise_for_status().json()
        assert reads == expected
        assert result['success']
        assert list(state.llm_history['shared'])[:-2] == history
        if expected:
            model.assert_not_awaited()
        else:
            assert state.control_memory.get(('shared', current)) is None
            assert model.call_args.kwargs['messages'][1:1+len(history)] == history
    finally:
        await provider.close_read_connection()


def test_timed_out_read_is_quarantined_until_late_response_is_discarded(cache):
    b = bridge(cache)
    sent = []
    b.client.publish = lambda *args, **kwargs: (sent.append(args[0]) or SimpleNamespace(rc=0))
    assert not b.read(['11'], timeout=.01)['success']
    def delayed(*args, **kwargs):
        sent.append(args[0])
        b.receive('jeedom/cmd/value/11', b'99', False)
        return SimpleNamespace(rc=0)
    b.client.publish = delayed
    blocked = b.read(['11'], timeout=.01)
    assert not blocked['success']
    assert len(sent) == 1
    b.receive('jeedom/cmd/value/11', b'99', False)
    def fresh(*args, **kwargs):
        sent.append(args[0])
        b.receive('jeedom/cmd/value/11', b'13', False)
        return SimpleNamespace(rc=0)
    b.client.publish = fresh
    result = b.read(['11'], timeout=.01)
    assert result['readings'][0]['value'] == 13
    assert len(sent) == 2


def test_retained_value_cannot_release_timeout_quarantine(cache):
    b = bridge(cache)
    b.client.publish = lambda *args, **kwargs: SimpleNamespace(rc=0)
    assert not b.read(['11'], timeout=.01)['success']
    b.receive('jeedom/cmd/value/11', b'99', True)
    b.client.publish = lambda *args, **kwargs: (b.receive('jeedom/cmd/value/11', b'99', False)
                                               or SimpleNamespace(rc=0))
    assert not b.read(['11'], timeout=.01)['success']


def test_timeout_quarantine_does_not_block_another_command(cache):
    b = bridge(cache)
    fresh = b.client.publish
    b.client.publish = lambda *args, **kwargs: SimpleNamespace(rc=0)
    assert not b.read(['11'], timeout=.01)['success']
    b.client.publish = fresh
    result = b.read(['21'], timeout=.01)
    assert result['success'] and result['readings'][0]['cmd_id'] == '21'
    assert not b.read(['11'], timeout=.01)['success']


def test_partial_timeout_only_quarantines_missing_command(cache):
    b = bridge(cache)
    fresh = b.client.publish
    def partial(topic, *args, **kwargs):
        if topic.endswith('/11'):
            return fresh(topic, *args, **kwargs)
        return SimpleNamespace(rc=0)
    b.client.publish = partial
    result = b.read(['11', '21'], timeout=.01)
    assert [r['status'] for r in result['readings']] == ['ok', 'unavailable']
    b.client.publish = fresh
    assert b.read(['11'], timeout=.01)['success']
    assert not b.read(['21'], timeout=.01)['success']
