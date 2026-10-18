import json
import time

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import pytest

from monitor.server import Feed, Tail, app
from monitor.ingress import active_socket, app as ingress


def event(name='Live.Request', **fields):
    return json.dumps(dict(event=name,time=time.time(),**fields))+'\n'


def test_tail_partial_rotation_and_untrusted_lines(tmp_path):
    path=tmp_path/'server.log';path.write_text(event(text='hello')+'not-json\n[]\n'+event('Unrelated.Secret'))
    tail=Tail(path); assert [r['data']['text'] for r in tail.read()]==['hello']
    with path.open('a') as f:f.write('{"event":"Live.Reply",')
    assert tail.read()==[]
    with path.open('a') as f:f.write('"time":1,"text":"reply"}\n')
    assert tail.read()[0]['data']['text']=='reply'
    path.unlink();path.write_text(event(text='rotated'))
    assert tail.read()[0]['data']['text']=='rotated'


def test_feed_gap_run_reset_client_join_and_path_scope(tmp_path):
    run=tmp_path/'run';run.mkdir();active=tmp_path/'active.json'
    active.write_text(json.dumps({'run_dir':str(run),'tools_mode':'log_only'}))
    (run/'server.log').write_text(event(session_id='client:kitchen',request_id='r1')+event('Live.Model',request_id='r1'))
    feed=Feed(active,tmp_path);feed.update();packet=feed.packet(-1,None)
    assert packet['type']=='snapshot'
    assert [e['client_id'] for e in packet['events']]==['kitchen','kitchen']
    with (run/'server.log').open('a') as f:f.write(event()*210)
    feed.update();assert feed.packet(2,'run')['type']=='snapshot'
    assert len(feed.packet(2,'run')['events'])==200
    active.write_text(json.dumps({'run_dir':'/outside'}));feed.update()
    assert feed.status['source']=='unavailable'


def test_http_and_websocket_are_readonly(tmp_path,monkeypatch):
    monkeypatch.setenv('ARIA_MONITOR_ACTIVE',str(tmp_path/'missing.json'))
    with TestClient(app) as client:
        response=client.get('/');assert response.status_code==200
        assert "default-src 'self'" in response.headers['content-security-policy']
        assert client.post('/').status_code==405
        assert client.post('/api/llm/process',json={'text':'execute'}).status_code==404
        with client.websocket_connect('/ws/events') as ws:
            assert ws.receive_json()['status']['source'] in {'waiting','unavailable'}
            ws.send_bytes(b'ignored');assert ws.receive_json()['type'] in {'snapshot','events'}
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/ws/events',headers={'origin':'http://untrusted'}):pass


def test_lan_ingress_only_exposes_audio_and_checks_run_scope(tmp_path,monkeypatch):
    pointer=tmp_path/'active.json';run=tmp_path/'run';run.mkdir()
    monkeypatch.setenv('ARIA_MONITOR_ACTIVE',str(pointer))
    monkeypatch.setenv('ARIA_MONITOR_RUN_ROOT',str(tmp_path))
    pointer.write_text(json.dumps({'run_dir':str(run),'tools_mode':'log_only'}))
    assert active_socket()==str(run/'aria.sock')
    pointer.write_text(json.dumps({'run_dir':'/elsewhere','tools_mode':'log_only'}))
    with pytest.raises(ValueError):active_socket()
    with TestClient(ingress) as client:
        assert client.get('/api/clients').status_code==404
        assert client.post('/api/llm/process',json={'text':'execute'}).status_code==404
