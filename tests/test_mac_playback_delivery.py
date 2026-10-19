"""Mac output adapter uses a fake subprocess; never launches afplay."""
import asyncio
import json
from types import SimpleNamespace
import httpx
import pytest
from scripts.e2e import mac_playback


@pytest.mark.asyncio
async def test_mac_acknowledges_player_lifecycle(tmp_path, monkeypatch):
    started = asyncio.Event()
    statuses, launches, requests = [], [], []
    polls = 0
    def respond(request):
        nonlocal polls
        path = request.url.path
        requests.append((request.method, path))
        if path == '/_live/config': return httpx.Response(200, json={'tools_mode': 'log_only', 'warmup_ready': True})
        if path == '/_live/player': return httpx.Response(200, json={'player_id': 'player'})
        if '/clip/' in path:
            status = json.loads(request.content)['status']; statuses.append(status)
            if status == 'started': started.set()
            return httpx.Response(200, json={'success': True})
        if path == '/_live/player/player': return httpx.Response(200, json={'success': True})
        if path.startswith('/tts/'): return httpx.Response(200, content=b'fake wav bytes')
        if path == '/_e2e/state':
            polls += 1
            if polls == 1: return httpx.Response(200, json={'next': 10, 'events': []})
            if polls == 2: return httpx.Response(200, json={'next': 11, 'events': [{
                'event': 'Live.Audio', 'url': 'http://aria/tts/' + 'a' * 32 + '.wav',
                'playback_id': 'b' * 32, 'request_id': 'request', 'time': 1.}]})
            return httpx.Response(410)
        raise AssertionError(path)
    client_type = httpx.AsyncClient
    monkeypatch.setattr(mac_playback.httpx, 'AsyncClient', lambda **kw: client_type(
        transport=httpx.MockTransport(respond), **kw))
    async def launch(*args):
        launches.append(args)
        process = SimpleNamespace(returncode=None)
        async def wait():
            await started.wait()
            process.returncode = 0
            return 0
        process.wait = wait
        process.terminate = lambda: None
        return process
    monkeypatch.setattr(mac_playback.asyncio, 'create_subprocess_exec', launch)
    with pytest.raises(httpx.HTTPStatusError):
        await asyncio.wait_for(mac_playback.listen('http://aria', tmp_path), 2)
    assert statuses == ['started', 'finished']
    assert len(launches) == 1
    assert ('DELETE', '/_live/player/player') in requests
    row = json.loads((tmp_path / 'playback.jsonl').read_text())
    assert row['request_id'] == 'request' and row['exit_code'] == 0
