"""Paired Apollo idle-residency benchmark; no microphone, playback, or actions.

Run sequential inference in two already-warmed, isolated live profiles. The
reference uses ARIA_LLM_KEEP_ALIVE_S=300; candidate uses -1. Both idle together.
Private output includes transcripts/inventory and per-process resource samples.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import httpx

from scripts.e2e.conversation_bench import build_cases, measure
from scripts.e2e.wake_bench import normalized_transcript


def process_sample(run):
    """Read only our namespace's inherited run marker; never log other environments."""
    rows=[]
    marker=('ARIA_E2E_RUN='+str(run)).encode()
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if marker not in (directory/'environ').read_bytes().split(b'\0'):
                continue
            args=(directory/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            stat=(directory/'stat').read_text().rsplit(')',1)[1].split()
            status=(directory/'status').read_text().splitlines()
            rss=next((int(s.split()[1]) for s in status if s.startswith('VmRSS:')),0)
            rows.append(dict(pid=int(directory.name),runner='runner --model' in args,
                             cpu_ticks=int(stat[11])+int(stat[12]),rss_kib=rss))
        except (OSError,ValueError,IndexError):
            continue
    return dict(time=time.time(),clock_ticks=os.sysconf('SC_CLK_TCK'),processes=rows)


def inference_events(run):
    events=[]
    for line in (run/'conversation.jsonl').read_text().splitlines():
        try:event=json.loads(line)
        except ValueError:continue
        if event.get('event') in {'Live.Model','Live.Prefill','Live.Warmup'}:
            events.append(event)
    # The raw model HTTP log independently detects calls that instrumentation misses.
    calls=sum('POST' in line and '/api/chat' in line
              for line in (run/'ollama.log').read_text().splitlines())
    return dict(events=len(events),chat_calls=calls)


async def run(args):
    if args.idle_seconds <= 300:
        raise ValueError('Test must exceed the previous five-minute idle interval')
    args.output.mkdir(parents=True,exist_ok=True)
    variants={'reference':args.reference_run,'candidate':args.candidate_run}
    report={'idle_seconds':args.idle_seconds,'mac_audio':False,'physical_actions':False,
            'runs':{k:str(v) for k,v in variants.items()},'idle':[],'cases':[],
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save():
        (args.output/'idle.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    fixtures=json.loads((args.fixtures/'manifest.json').read_text())
    for language in ['fr','en']:
        # Wait after both profiles' previous work/audio is complete. No generation
        # or keep-alive ping during this interval, only /proc and log reads.
        before={k:dict(resources=process_sample(v),inference=inference_events(v)) for k,v in variants.items()}
        began=time.monotonic();save()
        print(json.dumps(dict(event='idle_started',language=language,seconds=args.idle_seconds)),flush=True)
        while time.monotonic()-began < args.idle_seconds:
            await asyncio.sleep(min(30,args.idle_seconds-(time.monotonic()-began)))
            print(json.dumps(dict(event='idle_progress',language=language,elapsed=round(time.monotonic()-began))),flush=True)
        after={k:dict(resources=process_sample(v),inference=inference_events(v)) for k,v in variants.items()}
        idle=dict(language=language,elapsed=time.monotonic()-began,before=before,after=after)
        report['idle'].append(idle);save()
        assert all(before[k]['inference']==after[k]['inference'] for k in variants), 'Inference occurred during idle'
        for variant,root in variants.items():
            socket=str(root/'aria.sock');prefix='idle-'+uuid.uuid4().hex
            async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=socket),
                    base_url='http://localhost',timeout=180,trust_env=False) as client:
                config=(await client.get('/_live/config')).json()
                assert config['tools_mode']=='log_only' and config['warmup_ready'] and config['speaker_ready']
                # A real streamed PCM request is the first inference after idle.
                fixture=next(f for f in fixtures if f['language']==language and f['kind']=='inline')
                case=dict(id=language+'-idle-voice',language=language,kind='idle_voice',
                          text=fixture['text'],wav=fixture['path'],answer='84',
                          expected_commands=[],allow_search=False)
                row=await measure(client,socket,case,prefix)
                requests=[e for e in row['events'] if e['event']=='Live.Request']
                transitions=[e for e in row['events'] if e['event']=='ARIA.Conversation']
                # Literal ASR discrepancies stay recorded separately from response/
                # tool/state failures; name spelling repair is existing behavior.
                row['transcript_flags']=[f for f in row['flags'] if f=='transcript_mismatch']
                row['functional_flags']=[f for f in row['flags'] if f!='transcript_mismatch']
                row['normalized_transcript_equal']=bool(requests and normalized_transcript(requests[0]['text'])==normalized_transcript(case['text']))
                if len(transitions)!=1 or transitions[0]['action']!='activated':
                    row['functional_flags'].append('wake_transition')
                if not any(e['event']=='Live.Model' and e.get('tools_offered') for e in row['events']):
                    row['functional_flags'].append('missing_tool_ready_inference')
                row.update(variant=variant,config=config)
                report['cases'].append(row);save()
                print(json.dumps(dict(event='voice_result',variant=variant,language=language,
                      flags=row['functional_flags'],latency=row['latency'])),flush=True)
                # Full existing conversation/control checks, with activation on
                # the first request in each explicit session.
                cases=build_cases((await client.get('/api/inventory/snapshot')).json());sessions=set()
                for case in [c for c in cases if c['language']==language]:
                    if case['session'] not in sessions:
                        case['text']='Aria, '+case['text'];sessions.add(case['session'])
                    row=await measure(client,socket,case,prefix)
                    row.update(variant=variant,functional_flags=list(row['flags']))
                    report['cases'].append(row);save()
                    print(json.dumps(dict(event='case',variant=variant,id=case['id'],flags=row['flags'])),flush=True)
    report['functional_passed']=all(not r['functional_flags'] for r in report['cases'])
    report['strict_passed']=all(not r['flags'] for r in report['cases'])
    save()
    assert report['functional_passed'], 'Review functional failures; raw ASR flags retained separately'


if __name__=='__main__':
    os.umask(0o077)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--candidate-run',type=Path,required=True)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--idle-seconds',type=float,default=310)
    asyncio.run(run(parser.parse_args()))
