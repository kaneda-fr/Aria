"""Apollo-only conversation benchmark against an existing isolated log-only server.

No microphone, playback, model downloads, or real actions. Private reports contain
inventory names, transcripts, and speaker attribution. Automatic checks are guards;
review exact replies separately for semantic correctness and truthful feedback.
"""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
import unicodedata
import wave
from zoneinfo import ZoneInfo

import httpx
import websockets


def normalized(text):
    text = ''.join(c for c in unicodedata.normalize('NFKD', text.lower())
                   if not unicodedata.combining(c))
    return ' '.join(re.findall(r'\w+', text))


def build_cases(snapshot):
    devices = sorted(snapshot['devices'].values(), key=lambda d: d['name'])
    def choose(capabilities, exclude=()):
        for device in devices:
            if not device.get('enabled', True):
                continue
            groups = {cap: [c for c in device['commands']
                            if c['type'] == 'action' and c['capability'] == cap]
                      for cap in capabilities}
            if all(len(v) == 1 for v in groups.values()) and not any(
                    c['capability'] in exclude for c in device['commands']):
                return device, {k: v[0] for k, v in groups.items()}
        raise ValueError(f'No unambiguous inventory fixture for {capabilities}')
    lamp, lamp_commands = choose(['light_on', 'light_off'], ['light_slider'])
    dimmer, dim_commands = choose(['light_slider'])
    shutter, shutter_commands = choose(['flap_down'])
    cases = []
    for lang in ('fr', 'en'):
        french = lang == 'fr'
        dialogue = [
            ('fact', 'Quelle est la capitale de l’Italie ? Réponds seulement par le nom.' if french else
             'What is the capital of Italy? Answer with only its name.', 'rome'),
            ('arithmetic', 'Combien font douze fois sept ? Réponds seulement par le nombre.' if french else
             'What is twelve times seven? Answer with only the number.', '84'),
            ('remember', 'Mon mot secret pour cette conversation est abricot. Réponds seulement OK.' if french else
             'My secret word for this conversation is orchid. Reply only OK.', 'ok'),
            ('recall', 'Quel est mon mot secret ? Réponds seulement par ce mot.' if french else
             'What is my secret word? Answer with only that word.', 'abricot' if french else 'orchid'),
            ('capabilities', 'Nomme les deux outils de fonction dont tu disposes.' if french else
             'Name the two function tools available to you.', None),
            ('time', 'Quelle heure est-il maintenant ? Réponds au format HH:MM, heure de Zurich.' if french else
             'What time is it now? Answer in HH:MM, Zurich time.', None),
            ('topic_switch', 'Changeons de sujet : combien font neuf moins quatre ? Réponds seulement par le nombre.' if french else
             'Change of subject: what is nine minus four? Answer with only the number.', '5'),
        ]
        for kind, text, answer in dialogue:
            cases.append({'id': f'{lang}-{kind}', 'language': lang, 'kind': kind,
                          'session': f'chat-{lang}', 'text': text, 'answer': answer,
                          'expected_commands': [], 'allow_search': False})
        on = (f'Allume la lumière « {lamp["name"]} » dans la pièce « {lamp["room"]} ».' if french else
              f'Turn on the light named "{lamp["name"]}" in room "{lamp["room"]}".')
        off = 'Maintenant, éteins cette même lumière.' if french else 'Now turn that same light off.'
        dim = (f'Règle la lumière « {dimmer["name"]} » dans « {dimmer["room"]} » à 30 pour cent.' if french else
               f'Set the light "{dimmer["name"]}" in "{dimmer["room"]}" to 30 percent.')
        close = (f'Ferme le volet « {shutter["name"]} » dans « {shutter["room"]} ».' if french else
                 f'Close the shutter "{shutter["name"]}" in "{shutter["room"]}".')
        missing = ('Allume la lumière de la pièce « ARIA_TEST_ABSENT_ROOM ».' if french else
                   'Turn on the light in room "ARIA_TEST_ABSENT_ROOM".')
        ambiguous = 'Allume la lumière.' if french else 'Turn on the light.'
        for kind, text, command, value, payload in [
            ('on', on, lamp_commands['light_on']['cmd_id'], None, {}),
            ('off_followup', off, lamp_commands['light_off']['cmd_id'], None, {}),
            ('dim', dim, dim_commands['light_slider']['cmd_id'], 30, {'slider': 30}),
            ('shutter', close, shutter_commands['flap_down']['cmd_id'], None, {}),
            ('missing_room', missing, None, None, None),
            ('ambiguous', ambiguous, None, None, None),
        ]:
            expected = [] if command is None else [{'cmd_id': command, 'value': value, 'payload': payload}]
            cases.append({'id': f'{lang}-{kind}', 'language': lang, 'kind': kind,
                          'session': f'{lang}-light-dialogue' if kind in ('on', 'off_followup') else f'{lang}-{kind}',
                          'text': text, 'answer': None, 'expected_commands': expected, 'allow_search': True})
    return cases


def check_case(case, result, events):
    flags = []
    if not result.get('success'):
        flags.append('request_failed')
    if not result.get('spoken'):
        flags.append('speech_not_scheduled')
    requests = [e for e in events if e['event'] == 'Live.Request']
    if len(requests) != 1:
        flags.append('request_count_not_one')
    if not any(e['event'] == 'Live.Audio' for e in events):
        flags.append('audio_missing')
    if any(e.get('done_reason') == 'length' for e in events if e['event'] == 'Live.Model'):
        flags.append('generation_truncated')
    tools = [e for e in events if e['event'] == 'Live.Tool']
    if not case['allow_search'] and tools:
        flags.append('unexpected_tool_call')
    commands = [e for e in events if e['event'] == 'Live.Command']
    actual = [{'cmd_id': e['cmd_id'], 'value': e['value'],
               'payload': json.loads(e['payload']) if e['payload'] is not None else None} for e in commands]
    if actual != case['expected_commands']:
        flags.append('command_trace_mismatch')
    if any(not e['result'].get('success') for e in commands):
        flags.append('command_rejected')
    if any(e['result'].get('executed') is not False or not e['result'].get('dry_run') for e in commands):
        raise RuntimeError('Dry-run boundary violated')
    answer = result.get('response', '')
    if case.get('answer') is not None and normalized(answer) != normalized(case['answer']):
        flags.append('answer_mismatch')
    if case['kind'] == 'capabilities' and not all(name in answer for name in ['find_devices', 'execute_command']):
        flags.append('capability_list_incorrect')
    if case['kind'] == 'capabilities' and re.search(r'"(?:name|arguments|function)"\s*:', answer):
        flags.append('capability_reply_is_tool_json')
    if case['kind'] == 'time' and answer.strip() not in case['accepted_times']:
        flags.append('local_time_incorrect')
    return flags


def summarize(rows):
    values = sorted(x['latency']['request_to_audio_ready_s'] for x in rows
                    if x['latency'].get('request_to_audio_ready_s') is not None)
    return {'cases': len(rows), 'automatic_clear': sum(not x['flags'] for x in rows),
            'audio_completed': len(values),
            'p50_request_to_audio_s': statistics.median(values) if values else None,
            'sample_p95_request_to_audio_s': values[math.ceil(.95 * len(values))-1] if values else None}


async def collect(client, cursor, *, voice=False):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        state = (await client.get('/_e2e/state', params={'after': cursor})).json()
        events = state['events']
        replies = [e for e in events if e['event'] == 'Live.Reply']
        audio = [e for e in events if e['event'] == 'Live.Audio']
        if replies and (audio or not replies[-1].get('speech_scheduled')) and not state['speech_busy']:
            return state
        if voice and any(e['event'] == 'ARIA.Utterance.Final' and e.get('text_len') == 0 for e in events):
            return state
        await asyncio.sleep(.1)
    raise TimeoutError('No terminal response/audio outcome')


async def measure(client, socket, case, session_prefix):
    cursor = (await client.get('/_e2e/state')).json()['next']
    clock_start = datetime.now(ZoneInfo('Europe/Zurich'))
    started = time.monotonic()
    submitted_wall = time.time()
    voice_end = None
    voice_end_wall = None
    if case.get('wav'):
        path = Path(case['wav'])
        with wave.open(str(path)) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            pcm = wav.readframes(wav.getnframes())
        case['audio_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        async with websockets.unix_connect(socket, uri='ws://localhost/ws/asr') as ws:
            await ws.send(json.dumps({'type': 'start', 'sample_rate': 16000, 'channels': 1,
                                     'format': 'pcm_s16le', 'chunk_ms': 40, 'client_vad': True}))
            await ws.send(json.dumps({'type': 'utterance_start'}))
            for offset in range(0, len(pcm), 1280):
                await ws.send(pcm[offset:offset+1280])
                await asyncio.sleep(.04)
            voice_end = time.monotonic()
            voice_end_wall = time.time()
            await ws.send(json.dumps({'type': 'utterance_end'}))
            state = await collect(client, cursor, voice=True)
            observed_audio_end = time.monotonic()
            await ws.send(json.dumps({'type': 'stop'}))
        replies = [e for e in state['events'] if e['event'] == 'Live.Reply']
        result = ({'success': replies[-1].get('success'), 'response': replies[-1]['text'],
                   'spoken': replies[-1]['speech_scheduled']} if replies else
                  {'success': False, 'response': '', 'spoken': False})
    else:
        response = await client.post('/api/llm/process', json={
            'text': case['text'], 'session_id': session_prefix + '-' + case['session'],
            'speak_response': True, 'use_tools': True})
        result = response.json()
        state = await collect(client, cursor)
        observed_audio_end = time.monotonic()
    events = state['events']
    requests = [e for e in events if e['event'] == 'Live.Request']
    audio = [e for e in events if e['event'] == 'Live.Audio']
    models = [e for e in events if e['event'] == 'Live.Model']
    # Ground the clock oracle in the actual request interval, not a five-minute
    # tolerance that could accept a confidently wrong local time.
    replies = [e for e in events if e['event'] == 'Live.Reply']
    clock_bounds = [datetime.fromtimestamp(e['time'], ZoneInfo('Europe/Zurich'))
                    for e in requests[:1] + replies[-1:]] or [clock_start]
    case['accepted_times'] = sorted({instant.strftime('%H:%M') for instant in clock_bounds})
    flags = check_case(case, result, events)
    if case.get('wav') and (not requests or normalized(requests[-1]['text']) != normalized(case['text'])):
        flags.append('transcript_mismatch')
    status = [e for e in events if e['event'] == 'ARIA.Request.Status']
    accepted = next((e['time'] for e in status if e.get('status') == 'accepted'), None)
    running = next((e['time'] for e in status if e.get('status') == 'running'), None)
    latency = {'queue_wait_s': running-accepted if accepted is not None and running is not None else None,
               'accepted_to_first_audio_ready_s': audio[0]['time']-accepted if accepted is not None and audio else None,
               'submitted_to_first_audio_ready_s': audio[0]['time']-submitted_wall if audio and not case.get('wav') else None,
               'prefill_s': sum(e.get('seconds', 0) for e in events if e['event'] == 'ARIA.HistoryPrefill.Done'),
               'request_to_audio_ready_s': audio[-1]['time']-requests[0]['time'] if requests and audio else None,
               'request_to_first_audio_ready_s': audio[0]['time']-requests[0]['time'] if requests and audio else None,
               # Client and server run on Apollo, so these wall clocks share a host.
               'saved_pcm_end_to_first_audio_ready_s': audio[0]['time']-voice_end_wall if voice_end_wall and audio else None,
               # Observed durations below include waiting for speech_busy to clear.
               'observed_total_s': observed_audio_end-started,
               'saved_pcm_end_to_audio_observed_s': observed_audio_end-voice_end if voice_end and audio else None,
               'model_calls': len(models),
               'model_s': sum(e.get('total_duration') or 0 for e in models)/1e9,
               'prompt_s': sum(e.get('prompt_eval_duration') or 0 for e in models)/1e9,
               'generation_s': sum(e.get('eval_duration') or 0 for e in models)/1e9,
               'load_s': sum(e.get('load_duration') or 0 for e in models)/1e9,
               'generated_tokens': sum(e.get('eval_count') or 0 for e in models)}
    return {'case': case, 'result': result, 'events': events, 'latency': latency, 'flags': flags,
            'semantic_review': 'pending'}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--captures', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    socket = str(args.server_run / 'aria.sock')
    report = {'scope': 'Apollo shared text path and optional saved human PCM; no Mac activity or physical device actions',
              'server_run': str(args.server_run), 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cases': []}
    def save():
        report['summary'] = summarize(report['cases'])
        (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=socket),
                                base_url='http://localhost', timeout=300, trust_env=False) as client:
        config = (await client.get('/_live/config')).json()
        assert config['tools_mode'] == 'log_only' and config['speaker_ready'] and config['warmup_ready']
        snapshot = (await client.get('/api/inventory/snapshot')).json()
        report['config'] = config
        report['inventory_sha256'] = hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
        cases = build_cases(snapshot)
        if args.captures:
            for lang, text in [('en','The capital of France is Paris.'),('fr','La capitale de la France est Paris.')]:
                cases.append({'id': f'{lang}-saved-voice', 'language': lang, 'kind': 'saved_voice',
                              'text': text, 'wav': str(args.captures/lang/'segment-1.wav'),
                              'answer': None, 'expected_commands': [], 'allow_search': False})
        (args.output/'cases.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False))
        save()
        for case in cases:
            print(json.dumps({'starting':case['id']},ensure_ascii=False),flush=True)
            try:
                row = await measure(client, socket, case, args.output.name)
                report['cases'].append(row)
                save()
                print(json.dumps({'id':case['id'], 'flags':row['flags'], **row['latency']},ensure_ascii=False),flush=True)
            except BaseException as exc:
                report['interrupted_case'] = {'id':case['id'],'error':repr(exc)}
                save()
                raise


if __name__ == '__main__':
    asyncio.run(main())
