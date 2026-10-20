"""Offline server cache contention probes. Requires the isolated read fixture."""
import asyncio
import json
from pathlib import Path
import sys
import time
import uuid

import httpx
import websockets

from scripts.e2e.queue_bench import events_until


async def run(remote, output):
    socket = str(Path(remote) / 'aria.sock')
    report = {'scenarios': []}
    async with httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=socket), base_url='http://localhost',
        timeout=180, trust_env=False,
    ) as client:
        await events_until(client, 0, lambda s: not s['speech_busy'])
        config = (await client.get('/_live/config')).json()
        assert config['tools_mode'] == 'log_only' and config['history_max_messages'] == 6
        assert not (await client.get('/_live/playback')).json()['player_connected']
        inventory = (await client.get('/api/inventory/snapshot')).json()
        assert inventory['devices']['fixture1']['name'] == 'Lampe Canapé'
        report['config'] = config

        async def submit(text, session, speak=True):
            response = await client.post('/api/llm/process', json={
                'text': text, 'session_id': session, 'speak_response': speak, 'use_tools': True})
            response.raise_for_status()
            return response.json()

        async def audio(cursor, request_id):
            return await events_until(client, cursor, lambda s: any(
                e['event'] == 'Live.Audio' and e.get('request_id') == request_id
                for e in s['events']))

        async def idle():
            await events_until(client, 0, lambda s: not s['speech_busy'])
            while (await client.get('/_live/playback')).json()['input_blocked']:
                await asyncio.sleep(.05)

        def metrics(events, result):
            own = [e for e in events if e.get('request_id') == result['request_id']]
            at = lambda status: next(e['time'] for e in own
                                     if e['event'] == 'ARIA.Request.Status' and e.get('status') == status)
            first = next(e['time'] for e in own if e['event'] == 'Live.Audio')
            return {'queue_wait_s': at('running') - at('accepted'),
                    'accepted_to_first_audio_s': first - at('accepted'),
                    'prompt_s': sum(e.get('prompt_eval_duration') or 0 for e in own
                                    if e['event'] == 'Live.Model') / 1e9}

        for lang in ('fr', 'en'):
            fr = lang == 'fr'
            # Named connections send only the protocol handshake: no PCM/microphone.
            names = ['cache-test-' + uuid.uuid4().hex for _ in range(2)]
            async with websockets.unix_connect(socket, uri='ws://localhost/ws/asr') as wa, \
                       websockets.unix_connect(socket, uri='ws://localhost/ws/asr') as wb:
                for ws, name in zip((wa, wb), names):
                    await ws.send(json.dumps({'type': 'start', 'client_id': name,
                                              'sample_rate': 16000, 'channels': 1,
                                              'format': 'pcm_s16le', 'chunk_ms': 40,
                                              'client_vad': True}))
                await asyncio.sleep(.1)
                identities = (await client.get('/api/clients')).json()['clients']
                assert all(any(c['client_id'] == n and c['connected'] for c in identities) for n in names)
                sessions = ['client:' + n for n in names]
                for delay in (.03, .35):
                    for other in (False, True):
                        await idle()
                        cursor = (await client.get('/_e2e/state')).json()['next']
                        first = await submit(
                            'Aria, quelles commandes pour Lampe Lecture ?' if fr else
                            'Aria, what controls are available for Lampe Lecture?', sessions[0])
                        await audio(cursor, first['request_id'])
                        await asyncio.sleep(delay)
                        second = await submit(
                            'Aria, combien font douze fois sept ? Réponds seulement par le nombre.' if fr else
                            'Aria, what is twelve times seven? Answer with only the number.',
                            sessions[int(other)])
                        state = await audio(cursor, second['request_id'])
                        flags = []
                        if second.get('response', '').strip(' .') != '84':
                            flags.append('answer')
                        if any(e['event'] == 'Live.Command' for e in state['events']):
                            flags.append('unexpected_action')
                        report['scenarios'].append({
                            'kind': 'other_client' if other else 'same_client', 'language': lang,
                            'delay_after_audio_s': delay, 'first': first, 'second': second,
                            'latency': metrics(state['events'], second), 'events': state['events'],
                            'flags': flags})
                        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
                        print(lang, delay, other, flags, report['scenarios'][-1]['latency'], flush=True)

                # Each named client keeps a different resolved device/reference.
                await idle()
                setup_texts = (['Aria, quel est l’état de Lampe Canapé dans Salon Essai ?',
                                'Aria, quel est l’état de Lampe Lecture ?'] if fr else
                               ['Aria, what is the state of Lampe Canapé in Salon Essai?',
                                'Aria, what is the state of Lampe Lecture?'])
                for session, text in zip(sessions, setup_texts):
                    await submit(text, session, False)
                cursor = (await client.get('/_e2e/state')).json()['next']
                tasks = [asyncio.create_task(submit(
                    'Quel est son état ?' if fr else 'What is its state?', session, False))
                         for session in sessions]
                results = await asyncio.gather(*tasks)
                events = (await client.get('/_e2e/state', params={'after': cursor})).json()['events']
                flags = []
                for result, target in zip(results, ('fixture1s', 'fixture3s')):
                    reads = [r for e in events if e['event'] == 'ARIA.Inventory.Lookup'
                             and e.get('request_id') == result['request_id']
                             for r in e['result'].get('readings', [])]
                    if len(reads) != 1 or reads[0]['cmd_id'] != target:
                        flags.append('reference_leak')
                report['scenarios'].append({'kind': 'concurrent_references', 'language': lang,
                                            'results': results, 'events': events, 'flags': flags})
                Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
                for ws in (wa, wb):
                    await ws.send(json.dumps({'type': 'stop'}))
        await idle()
        report['final_playback'] = (await client.get('/_live/playback')).json()
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(run(sys.argv[1], sys.argv[2]))
