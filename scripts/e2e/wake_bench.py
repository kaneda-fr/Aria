"""Offline Apollo wake/follow-up/expiry checks; captured WAV and log-only tools."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import uuid
import wave

import httpx
import websockets

from scripts.e2e.conversation_bench import build_cases, normalized
from scripts.e2e.queue_bench import scenario


def normalized_transcript(text):
    # Only equivalent number/operator spellings in the fixed arithmetic fixtures.
    # Preserve lexical errors such as "de" for "deux", and never normalize names.
    words = {'two': '2', 'deux': '2', 'twelve': '12', 'douze': '12',
             'seven': '7', 'sept': '7', 'times': 'x', 'fois': 'x'}
    return ' '.join(words.get(word, word) for word in normalized(text.replace('×', ' x ')).split())


async def run(args):
    socket = str(args.server_run / 'aria.sock')
    report = {'cases': [], 'queue': [], 'mac_audio': False, 'physical_actions': False}
    prefix = 'wake-' + uuid.uuid4().hex
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=socket),
                                base_url='http://localhost', timeout=180, trust_env=False) as client:
        config = (await client.get('/_live/config')).json()
        assert config['tools_mode'] == 'log_only' and config['warmup_ready'] and config['speaker_ready']
        report['config'] = config
        snapshot = (await client.get('/api/inventory/snapshot')).json()
        controls = {case['id']: case for case in build_cases(snapshot)}

        async def collect(cursor):
            async with asyncio.timeout(120):
                while True:
                    state = (await client.get('/_e2e/state', params={'after': cursor})).json()
                    # Final ASR precedes model/TTS; it is not request completion.
                    statuses = [e for e in state['events'] if e['event'] == 'ARIA.Request.Status']
                    terminal = any(e.get('status') in {'completed', 'failed', 'rejected', 'expired', 'cancelled'}
                                   for e in statuses)
                    if not statuses:
                        terminal = any(e['event'] == 'ARIA.Utterance.Final' for e in state['events'])
                    if terminal and not state['speech_busy']:
                        return state['events']
                    await asyncio.sleep(.05)

        def check(cid, result, events, action, answer, expected, voice_end=None):
            flags = []
            transitions = [e for e in events if e['event'] == 'ARIA.Conversation']
            if len(transitions) != 1 or transitions[0]['action'] != action:
                flags.append('conversation_transition')
            if transitions and transitions[0]['idle_timeout_s'] != 30:
                flags.append('not_default_timeout')
            commands = [e for e in events if e['event'] == 'Live.Command']
            actual = [{'cmd_id': e['cmd_id'], 'value': e['value'],
                       'payload': json.loads(e['payload']) if e['payload'] is not None else None} for e in commands]
            if actual != expected:
                flags.append('command_trace')
            assert all(e['result'].get('dry_run') and e['result'].get('executed') is False for e in commands)
            if answer is not None and normalized(result.get('response', '')) != normalized(answer):
                flags.append('reply')
            models = [e for e in events if e['event'] == 'Live.Model']
            audio = [e for e in events if e['event'] == 'Live.Audio']
            if action == 'ignored' and (models or audio or commands or result.get('response')):
                flags.append('inactive_side_effect')
            if action == 'closed' and models:
                flags.append('goodbye_used_model')
            accepted = next((e['time'] for e in events if e.get('status') == 'accepted'), None)
            row = dict(id=cid, result=result, flags=flags, events=events,
                       accepted_to_audio_s=audio[0]['time']-accepted if accepted and audio else None,
                       pcm_end_to_audio_s=audio[0]['time']-voice_end if voice_end and audio else None)
            report['cases'].append(row)
            save()
            print(json.dumps({k:v for k,v in row.items() if k != 'events'}, ensure_ascii=False), flush=True)
            return row

        async def text(cid, session, utterance, action, answer=None, expected=()):
            cursor = (await client.get('/_e2e/state')).json()['next']
            response = await client.post('/api/llm/process', json=dict(
                text=utterance, session_id=prefix+'-'+session, speak_response=True, use_tools=True))
            response.raise_for_status()
            events = await collect(cursor)
            row = check(cid, response.json(), events, action, answer, list(expected))
            requests = [e for e in events if e['event'] == 'Live.Request']
            assert len(requests) == 1 and requests[0]['text'] == utterance
            return row

        if not args.voice_only:
            for lang in ['fr', 'en']:
                fr = lang == 'fr'
                session = lang+'-text'
                wake = 'Bonjour Aria' if fr else 'Hello Aria'
                bye = 'Merci, au revoir Aria' if fr else 'Thanks, goodbye Aria'
                on, off = controls[lang+'-on'], controls[lang+'-off_followup']
                remember, recall = controls[lang+'-remember'], controls[lang+'-recall']
                await text(lang+'-asleep-control', session, on['text'], 'ignored', '')
                await text(lang+'-mention', session, 'Qui est Aria ?' if fr else 'Who is Aria?', 'ignored', '')
                await text(lang+'-name-only', session, wake, 'activated', 'Oui' if fr else 'Yes')
                await text(lang+'-remember', session, remember['text'], 'followup', 'OK')
                await text(lang+'-recall', session, recall['text'], 'followup', recall['answer'])
                await text(lang+'-other-session', lang+'-other', recall['text'], 'ignored', '')
                await text(lang+'-on', session, on['text'], 'followup', expected=on['expected_commands'])
                await text(lang+'-off', session, off['text'], 'followup', expected=off['expected_commands'])
                await text(lang+'-goodbye', session, bye, 'closed', 'Au revoir' if fr else 'Goodbye')
                await text(lang+'-after-goodbye', session, on['text'], 'ignored', '')
                row = await text(lang+'-reset-reference', session, 'Aria, '+off['text'], 'activated')
                assert row['result'].get('control_status') == 'no_reference'
                await text(lang+'-close-again', session, bye, 'closed', 'Au revoir' if fr else 'Goodbye')
                arithmetic = controls[lang+'-arithmetic']
                await text(lang+'-inline', session, 'Aria, '+arithmetic['text'], 'activated', '84')

            # Both languages use the actual default 30-second window, measured after
            # capture-sink playback completion. Wait once for the two independent sessions.
            for lang in ['fr', 'en']:
                await text(lang+'-idle-setup', lang+'-idle', 'Aria, '+controls[lang+'-remember']['text'], 'activated', 'OK')
            await asyncio.sleep(31)
            for lang in ['fr', 'en']:
                await text(lang+'-expired', lang+'-idle', controls[lang+'-recall']['text'], 'ignored', '')
                row = await text(lang+'-new-history', lang+'-idle', 'Aria, '+controls[lang+'-arithmetic']['text'], 'activated', '84')
                models = [e for e in row['events'] if e['event'] == 'Live.Model']
                assert models and models[0]['history_messages'] == 0

            # Wake-enabled adaptation of the unchanged overlap/overload benchmark.
            class WakingClient:
                def __init__(self): self.sessions = set()
                async def get(self, *a, **kw): return await client.get(*a, **kw)
                async def post(self, url, *, json):
                    body = dict(json)
                    if body['session_id'] not in self.sessions:
                        body['text'] = 'Aria, '+body['text']
                        self.sessions.add(body['session_id'])
                    return await client.post(url, json=body)
            waking = WakingClient()
            for lang in ['fr', 'en']:
                report['queue'].append(await scenario(waking, lang, 0))
                save()

        manifest = json.loads((args.fixtures/'manifest.json').read_text())
        report['fixtures'] = manifest
        for lang in ['fr', 'en']:
            fixtures = {f['kind']: f for f in manifest if f['language'] == lang}
            async with websockets.unix_connect(socket, uri='ws://localhost/ws/asr', max_queue=None) as ws:
                await ws.send(json.dumps(dict(type='start', sample_rate=16000, channels=1,
                                             format='pcm_s16le', chunk_ms=40, client_vad=True)))
                for kind, action, answer in [
                    ('question','ignored',''), ('wake','activated','Oui' if lang=='fr' else 'Yes'),
                    ('question','followup','4'), ('goodbye','closed','Au revoir' if lang=='fr' else 'Goodbye'),
                    ('question','ignored',''), ('inline','activated','84')]:
                    cursor = (await client.get('/_e2e/state')).json()['next']
                    with wave.open(fixtures[kind]['path']) as wav:
                        assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth()) == (16000,1,2)
                        pcm = wav.readframes(wav.getnframes())
                    await ws.send(json.dumps({'type':'utterance_start'}))
                    for pos in range(0,len(pcm),1280):
                        await ws.send(pcm[pos:pos+1280])
                        await asyncio.sleep(.04)
                    ended = time.time()
                    await ws.send(json.dumps({'type':'utterance_end'}))
                    events = await collect(cursor)
                    replies = [e for e in events if e['event']=='Live.Reply']
                    result = {'response':replies[-1]['text'] if replies else ''}
                    row = check(lang+'-voice-'+kind+'-'+action, result, events, action, answer, [], ended)
                    requests = [e for e in events if e['event']=='Live.Request']
                    if not requests or normalized_transcript(requests[0]['text']) != normalized_transcript(fixtures[kind]['text']):
                        row['flags'].append('transcript_mismatch')
                    save()
                await ws.send(json.dumps({'type':'stop'}))
        report['passed'] = all(not r['flags'] for r in report['cases']+report['queue'])
        save()
        assert report['passed'], 'Inspect per-case wake report'


if __name__ == '__main__':
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-run', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--voice-only', action='store_true', help='Replay another input voice without repeating text/queue checks')
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
