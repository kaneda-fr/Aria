"""The live action boundary logs intent without constructing a broker client."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.plugins.actions.provider import ActionsProvider
from scripts.e2e.live_tools import LogOnlyActions


@pytest.mark.parametrize('cmd_id,value,error,payload', [
    ('slider', 30, None, '{"slider": 30}'),
    ('slider', 130, 'Value must be between 0 and 100', None),
    ('missing', 30, 'unknown_command', None),
])
def test_log_only_validates_and_never_publishes(monkeypatch, cmd_id, value, error, payload):
    def forbidden(*args, **kwargs):
        raise AssertionError('Real action lifecycle or execution reached')
    monkeypatch.setattr(ActionsProvider, 'start', forbidden)
    monkeypatch.setattr(ActionsProvider, 'stop', forbidden)
    monkeypatch.setattr(ActionsProvider, 'execute_command', forbidden)
    command = SimpleNamespace(execution={'args': {
        'value_required': True, 'value_type': 'int', 'range': [0, 100], 'payload_template': 'slider'}})
    log = Mock()
    actions = LogOnlyActions('actions', {
        'inventory': SimpleNamespace(indexer=SimpleNamespace(
            get_command=lambda identifier: command if identifier == 'slider' else None)),
        'log': log, 'request_id': lambda: 'request-1'})
    actions.start()
    assert actions.is_started and actions._client is None
    result = actions.execute_command(cmd_id, value, {'args': {}})
    assert result['success'] == (error is None)
    assert result['error'] == error
    assert result['dry_run'] and not result['executed']
    assert log.call_args.kwargs['request_id'] == 'request-1'
    assert log.call_args.kwargs['payload'] == payload
    actions.stop()
    assert not actions.is_started and actions._client is None
