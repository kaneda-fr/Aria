import json
import httpx
from fastapi.testclient import TestClient
import monitor.server as monitor


def test_monitor_mute_has_same_origin_run_and_payload_guards(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();pointer=tmp_path/'active.json'
    pointer.write_text(json.dumps({'run_dir':str(run),'tools_mode':'log_only'}))
    monkeypatch.setenv('ARIA_MONITOR_ACTIVE',str(pointer))
    monkeypatch.setenv('ARIA_MONITOR_RUN_ROOT',str(tmp_path))
    forwarded=[]
    def backend(request):
        if request.method=='GET':return httpx.Response(200,json={'clients':[]})
        forwarded.append((request.url.path,json.loads(request.content)))
        return httpx.Response(200,json={'client_id':'kitchen','muted':json.loads(request.content)['muted'],'mute_remaining_s':300})
    original=httpx.AsyncClient
    monkeypatch.setattr(monitor.httpx,'AsyncClient',lambda **kw:original(**{**kw,'transport':httpx.MockTransport(backend)}))
    headers={'origin':'http://testserver','x-aria-control':'mute'}
    body={'muted':True,'run_id':'run'}
    with TestClient(monitor.app) as client:
        url='/api/clients/kitchen/mute'
        assert client.post(url,json=body).status_code==403
        assert client.post(url,json=body,headers={**headers,'origin':'https://elsewhere'}).status_code==403
        assert client.post(url,json={**body,'run_id':'old'},headers=headers).status_code==409
        assert client.post(url,json={**body,'muted':'false'},headers=headers).status_code==422
        assert forwarded==[]
        assert client.post(url,json=body,headers=headers).json()['muted'] is True
        assert client.post(url,json={**body,'muted':False},headers=headers).json()['muted'] is False
        assert forwarded==[('/api/clients/kitchen/mute',{'muted':True}),('/api/clients/kitchen/mute',{'muted':False})]
