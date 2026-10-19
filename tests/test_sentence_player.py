"""The output-only Mac adapter avoids a fixed 100 ms inter-clip wait."""
import asyncio
import json
from types import SimpleNamespace
import httpx
import pytest
from scripts.e2e import mac_playback


@pytest.mark.asyncio
async def test_fast_poll_is_bounded_to_transition_then_returns_to_idle(tmp_path, monkeypatch):
    clock=[100.];sleeps=[];polls=0;acks=[]
    def respond(request):
        nonlocal polls
        path=request.url.path
        if path=='/_live/config':return httpx.Response(200,json={'tools_mode':'log_only','warmup_ready':True})
        if path=='/_live/player':return httpx.Response(200,json={'player_id':'player'})
        if '/clip/' in path:
            acks.append(json.loads(request.content)['status']);return httpx.Response(200,json={})
        if path=='/_live/player/player':return httpx.Response(200,json={})
        if path.startswith('/tts/'):return httpx.Response(200,content=b'fake')
        if path=='/_e2e/state':
            polls+=1
            if polls>33:return httpx.Response(410)
            events=([{'event':'Live.Audio','url':'http://aria/tts/'+'a'*32+'.wav',
                'playback_id':'b'*32,'time':clock[0]}] if polls==2 else [])
            return httpx.Response(200,json={'next':polls,'events':events})
        raise AssertionError(path)
    real_client=httpx.AsyncClient
    monkeypatch.setattr(mac_playback.httpx,'AsyncClient',lambda **kw:real_client(transport=httpx.MockTransport(respond),**kw))
    monkeypatch.setattr(mac_playback,'time',SimpleNamespace(monotonic=lambda:clock[0],time=lambda:clock[0]))
    real_sleep = asyncio.sleep
    async def sleep(delay):
        if delay == 5:
            return await real_sleep(delay)  # Independent lease task, canceled on exit.
        sleeps.append(delay);clock[0]+=delay
        await real_sleep(0)
    monkeypatch.setattr(mac_playback.asyncio,'sleep',sleep)
    async def launch(*args):
        proc=SimpleNamespace(returncode=None,terminate=lambda:None)
        async def wait():proc.returncode=0;return 0
        proc.wait=wait
        return proc
    monkeypatch.setattr(mac_playback.asyncio,'create_subprocess_exec',launch)
    with pytest.raises(httpx.HTTPStatusError):
        await asyncio.wait_for(mac_playback.listen('http://aria',tmp_path),2)
    assert acks==['started','finished']
    assert sleeps[0]==.01
    assert 24<=sleeps.count(.01)<=26
    assert sleeps[-1]==.1
