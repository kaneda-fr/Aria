"""Expired player leases and canceled events, using fake audio processes."""
import asyncio
import json
from types import SimpleNamespace
import httpx
import pytest
from app.speech_output.playback import PlaybackBroker
from scripts.e2e import mac_playback


@pytest.mark.asyncio
async def test_long_clip_releases_when_player_lease_expires():
    broker = PlaybackBroker()
    broker.register()
    broker.deadline = broker.clock() + .02
    _, task = broker.begin(60.)
    with pytest.raises(RuntimeError, match='player_lease_expired'):
        await asyncio.wait_for(task, .3)
    assert not broker.pending


@pytest.mark.asyncio
async def test_heartbeat_renews_lease_during_long_clip():
    broker = PlaybackBroker()
    player = broker.register()
    broker.deadline = broker.clock() + .02
    clip, task = broker.begin(60.)
    await asyncio.sleep(.01)
    broker.heartbeat(player)
    await asyncio.sleep(.03)
    assert not task.done()
    broker.ack(player, clip, 'started')
    broker.ack(player, clip, 'finished')
    await task
    assert not broker.pending


@pytest.mark.asyncio
@pytest.mark.parametrize('canceled_at', ['started', 'finished'])
async def test_mac_skips_canceled_ticket_and_still_plays_next(tmp_path, monkeypatch, canceled_at):
    polls = 0
    accepted_starts, launches = [], []
    def respond(request):
        nonlocal polls
        path = request.url.path
        if path == '/_live/config': return httpx.Response(200,json={'tools_mode':'log_only','warmup_ready':True})
        if path == '/_live/player': return httpx.Response(200,json={'player_id':'player'})
        if path == '/_live/player/player': return httpx.Response(200,json={})
        if '/clip/' in path:
            status = json.loads(request.content)['status']
            clip = path.rsplit('/',1)[-1]
            if clip == 'b'*32 and status == canceled_at: return httpx.Response(409,json={'detail':'unknown_playback'})
            if status == 'started': accepted_starts.append(clip)
            return httpx.Response(200,json={})
        if path.startswith('/tts/'): return httpx.Response(200,content=b'fake')
        if path == '/_e2e/state':
            polls += 1
            if polls > 2: return httpx.Response(410)
            events = [] if polls == 1 else [{'event':'Live.Audio','url':'http://aria/tts/'+'a'*32+'.wav','playback_id':c*32,'time':0.} for c in ('b','c')]
            return httpx.Response(200,json={'next':polls,'events':events})
        raise AssertionError(path)
    original = httpx.AsyncClient
    monkeypatch.setattr(mac_playback.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw))
    async def launch(*args):
        assert len(accepted_starts) > len(launches), 'Validate the ticket before launching audio'
        launches.append(args)
        proc = SimpleNamespace(returncode=None,terminate=lambda:None)
        async def wait(): proc.returncode=0; return 0
        proc.wait=wait
        return proc
    monkeypatch.setattr(mac_playback.asyncio,'create_subprocess_exec',launch)
    with pytest.raises(httpx.HTTPStatusError) as end:
        await asyncio.wait_for(mac_playback.listen('http://aria',tmp_path),2)
    assert end.value.request.url.path == '/_e2e/state'
    assert len(launches) == (1 if canceled_at == 'started' else 2)
    rows = [json.loads(line) for line in (tmp_path/'playback.jsonl').read_text().splitlines()]
    assert rows[-1]['playback_id'] == 'c'*32
