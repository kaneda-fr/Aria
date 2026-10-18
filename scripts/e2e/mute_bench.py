"""Apollo named-client mute checks, using text/saved PCM only; never opens hardware."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import wave
import httpx
import websockets
from scripts.e2e.conversation_bench import collect


async def connect(socket, name):
    ws = await websockets.unix_connect(socket, uri='ws://localhost/ws/asr')
    await ws.send(json.dumps(dict(type='start',sample_rate=16000,channels=1,
                                 format='pcm_s16le',chunk_ms=40,client_vad=True,client_id=name)))
    ack = json.loads(await asyncio.wait_for(ws.recv(),3))
    assert ack['type']=='session_started' and ack['client_id']==name
    return ws, ack


async def run(args):
    socket=str(args.server_run/'aria.sock'); rows=[];report={'cases':rows}
    def save(): args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=socket),
                                base_url='http://aria',timeout=120,trust_env=False) as client:
        report['config']=(await client.get('/_live/config')).json()
        assert report['config']['tools_mode']=='log_only' and report['config']['warmup_ready']
        async def status(name):
            return next(c for c in (await client.get('/api/clients')).json()['clients'] if c['client_id']==name)
        async def text(name, words):
            start=time.monotonic()
            result=(await client.post('/api/llm/process',json=dict(text=words,session_id='client:'+name,
                speak_response=False,use_tools=True))).json()
            rows.append(dict(client_id=name,input=words,result=result,seconds=time.monotonic()-start));save()
            return result
        for lang,request,yes,no in [('en','Aria, mute yourself for 2 seconds','yes','no'),
                                   ('fr','Aria, mets-toi en sourdine pendant deux secondes','oui','non')]:
            name='bench-mute-'+lang;ws,first=await connect(socket,name)
            try:
                assert (await text(name,request))['mute_status']=='confirmation_required'
                assert not (await status(name))['muted']
                assert (await text(name,no))['mute_status']=='cancelled'
                assert (await text(name,request))['mute_status']=='confirmation_required'
                assert (await text(name,yes))['mute_status']=='muted'
                assert (await status(name))['muted']
                await ws.close();ws,second=await connect(socket,name)
                assert first['session_id']==second['session_id'] and first['connection_id']!=second['connection_id']
                assert (await status(name))['muted']
                await asyncio.sleep(2.1)
                assert not (await status(name))['muted']
                rows.append(dict(client_id=name,reconnect_and_short_expiry='pass'));save()
            finally: await ws.close()

        if args.fixtures:
            for lang in ('en','fr'):
                name='bench-voice-mute-'+lang;ws,_=await connect(socket,name)
                try:
                    for kind in ('request','confirm'):
                        cursor=(await client.get('/_e2e/state')).json()['next']
                        with wave.open(str(args.fixtures/f'{lang}-{kind}.wav')) as wav:
                            assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())==(16000,1,2)
                            pcm=wav.readframes(wav.getnframes())
                        await ws.send(json.dumps({'type':'utterance_start'}))
                        for offset in range(0,len(pcm),1280):
                            await ws.send(pcm[offset:offset+1280]);await asyncio.sleep(.04)
                        end=time.time();await ws.send(json.dumps({'type':'utterance_end'}))
                        state=await collect(client,cursor,voice=True)
                        events=state['events'];audio=[e for e in events if e['event']=='Live.Audio']
                        command=[e for e in events if e['event']=='ARIA.Session.MuteCommand']
                        row=dict(client_id=name,kind=kind,events=events,
                                 speech_end_to_audio_s=audio[0]['time']-end if audio else None,
                                 passed=bool(command and command[-1]['status']==('confirmation_required' if kind=='request' else 'muted')))
                        rows.append(row);save()
                        if not row['passed']: break
                    await client.post(f'/api/clients/{name}/mute',json={'muted':False})
                finally: await ws.close()

        if args.skip_long_timer:
            report['five_minute']={'status':'skipped_explicitly'};save()
            failures=[r['client_id'] for r in rows if r.get('passed') is False]
            print(json.dumps({'voice_failures':failures}),flush=True)
            assert not failures, failures
            return
        # Real five-minute expiry, independent of browser state or model activity.
        name='bench-five-minute';ws,_=await connect(socket,name)
        try:
            started=time.monotonic()
            result=(await client.post(f'/api/clients/{name}/mute',json={'muted':True})).json()
            assert 299 <= result['mute_remaining_s'] <= 300
            report['five_minute']={'started':time.time(),'status':'waiting'};save()
            while time.monotonic()-started < 305:
                row=await status(name);elapsed=time.monotonic()-started
                if not row['muted']:
                    assert elapsed >= 299.5
                    report['five_minute'].update(status='pass',observed_expiry_s=elapsed);save();break
                await asyncio.sleep(1)
            else:raise AssertionError('Mute did not expire')
        finally: await ws.close()
    print(json.dumps({'cases':len(rows),'voice_failures':[r['client_id'] for r in rows if r.get('passed') is False],
                      'five_minute':report['five_minute']}),flush=True)
    assert all(r.get('passed',True) for r in rows), 'Saved-voice mute flow failed; inspect report'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--server-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--fixtures',type=Path)
    p.add_argument('--skip-long-timer',action='store_true')
    asyncio.run(run(p.parse_args()))
