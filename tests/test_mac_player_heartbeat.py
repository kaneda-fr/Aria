"""Player lease continues through delayed network reads; audio is mocked."""
import asyncio
from types import SimpleNamespace
import httpx
import pytest
from scripts.e2e import mac_playback


@pytest.mark.asyncio
@pytest.mark.parametrize("heartbeat_fails", [False, True])
async def test_heartbeat_runs_during_state_and_wav_reads_and_stops_on_exit(tmp_path, monkeypatch, heartbeat_fails):
    state_renewed = asyncio.Event()
    wav_renewed = asyncio.Event()
    phase = ['idle']
    polls = 0
    renewals = []
    launches = []
    real_sleep = asyncio.sleep
    async def sleep(delay):
        await real_sleep(.005 if delay == 5 else delay)
    monkeypatch.setattr(mac_playback.asyncio, 'sleep', sleep)
    class DelayedWav(httpx.AsyncByteStream):
        async def __aiter__(self):
            phase[0] = 'wav'
            await asyncio.wait_for(wav_renewed.wait(), .2)
            yield b'fake wav'
    async def respond(request):
        nonlocal polls
        path = request.url.path
        if path == '/_live/config': return httpx.Response(200,json={'tools_mode':'log_only','warmup_ready':True})
        if path == '/_live/player': return httpx.Response(200,json={'player_id':'player'})
        if path == '/_live/player/player':
            if request.method == 'PUT':
                renewals.append(phase[0])
                if phase[0] == 'state': state_renewed.set()
                if phase[0] == 'wav':
                    wav_renewed.set()
                    if heartbeat_fails: return httpx.Response(409,json={'detail':'expired'})
            return httpx.Response(200,json={})
        if '/clip/' in path: return httpx.Response(200,json={})
        if path.startswith('/tts/'): return httpx.Response(200,stream=DelayedWav())
        if path == '/_e2e/state':
            polls += 1
            if polls > 2: return httpx.Response(410)
            if polls == 1: return httpx.Response(200,json={'next':1,'events':[]})
            phase[0] = 'state'
            await asyncio.wait_for(state_renewed.wait(), .2)
            return httpx.Response(200,json={'next':2,'events':[{'event':'Live.Audio',
                'url':'http://aria/tts/'+'a'*32+'.wav','playback_id':'b'*32,'time':0.}]})
        raise AssertionError(path)
    original = httpx.AsyncClient
    monkeypatch.setattr(mac_playback.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(respond),**kw))
    async def launch(*args):
        launches.append(args)
        proc=SimpleNamespace(returncode=None,terminate=lambda:None)
        async def wait(): proc.returncode=0; return 0
        proc.wait=wait
        return proc
    monkeypatch.setattr(mac_playback.asyncio,'create_subprocess_exec',launch)
    with pytest.raises(httpx.HTTPStatusError) as end:
        await asyncio.wait_for(mac_playback.listen('http://aria',tmp_path),2)
    assert end.value.request.url.path == ('/_live/player/player' if heartbeat_fails else '/_e2e/state')
    assert state_renewed.is_set() and wav_renewed.is_set()
    assert len(launches) == (0 if heartbeat_fails else 1)
    count = len(renewals)
    await real_sleep(.03)
    assert len(renewals) == count
