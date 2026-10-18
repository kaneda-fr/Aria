import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.client_sessions import ClientSessions


def test_clear_offline_preserves_connected_mute_and_pending_work():
    clients = ClientSessions()
    for name in ['online', 'offline', 'muted']:
        clients.claim(name, name+'-socket')
    clients.set_muted('muted', True)
    deadline = clients.known['muted']['_mute_deadline']
    clients.release('offline', 'offline-socket')
    clients.release('muted', 'muted-socket')
    clients.request(dict(request_id='pending', session_id='client:offline', status='accepted'))
    assert clients.clear_offline() == ['offline', 'muted']
    assert clients.clear_offline() == []
    assert [c['client_id'] for c in clients.snapshot()] == ['online']
    assert clients.requests['pending'] == ('offline', 'accepted')
    assert clients.claim('muted', 'new-socket') == 'client:muted'
    assert clients.known['muted']['_mute_deadline'] == deadline
    assert clients.is_muted('muted')
    assert {c['client_id'] for c in clients.snapshot()} == {'online', 'muted'}
    clients.request(dict(request_id='pending', session_id='client:offline', status='completed'))
    assert not clients.requests


@pytest.mark.asyncio
async def test_cleanup_endpoint_returns_authoritative_visible_snapshot(monkeypatch):
    import app.main as runtime
    registry = ClientSessions()
    registry.claim('disconnected', 'one')
    registry.release('disconnected', 'one')
    registry.claim('connected', 'two')
    monkeypatch.setattr(runtime.app.state, 'client_sessions', registry, raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app), base_url='http://aria') as client:
        response = await client.post('/api/clients/clear-offline')
        assert response.status_code == 200
        assert response.json()['cleared'] == ['disconnected']
        assert [c['client_id'] for c in response.json()['clients']] == ['connected']


def test_monitor_cleanup_checks_origin_and_active_run(tmp_path, monkeypatch):
    import monitor.server as monitor
    run = tmp_path/'run'; run.mkdir()
    pointer = tmp_path/'active.json'
    pointer.write_text(json.dumps(dict(run_dir=str(run), tools_mode='log_only')))
    monkeypatch.setenv('ARIA_MONITOR_ACTIVE', str(pointer))
    monkeypatch.setenv('ARIA_MONITOR_RUN_ROOT', str(tmp_path))
    forwarded = []
    def backend(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'clients': []})
        forwarded.append(request.url.path)
        return httpx.Response(200, json={'cleared': ['old'], 'clients': []})
    original = httpx.AsyncClient
    monkeypatch.setattr(monitor.httpx, 'AsyncClient', lambda **kw: original(
        **{**kw, 'transport': httpx.MockTransport(backend)}))
    with TestClient(monitor.app) as client:
        url = '/api/clients/clear-offline'
        headers = {'origin': 'http://testserver', 'x-aria-control': 'clear-offline'}
        assert client.post(url, json={'run_id':'run'}).status_code == 403
        assert client.post(url, json={'run_id':'old'}, headers=headers).status_code == 409
        assert not forwarded
        response = client.post(url, json={'run_id':'run'}, headers=headers)
        assert response.json() == {'cleared': ['old'], 'clients': []}
        assert forwarded == [url]
