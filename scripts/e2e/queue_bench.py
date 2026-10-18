"""Apollo-only overlap probe: prefill cancellation, FIFO wait, and bounded overload.

No microphone or playback device. First overlapping request is text-only so its
audio cannot be replaced by the second request; the queued request synthesizes WAV.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid

import httpx


async def events_until(client, cursor, predicate):
    async with asyncio.timeout(90):
        while True:
            state = (await client.get('/_e2e/state', params={'after': cursor})).json()
            if predicate(state):
                return state
            await asyncio.sleep(.02)


async def scenario(client, language, repetition):
    session = 'queue-' + uuid.uuid4().hex
    fr = language == 'fr'
    def body(text, speak=False):
        return dict(text=text, session_id=session, speak_response=speak, use_tools=True)
    await client.post('/api/llm/process', json=body(
        'Aria, Combien font deux plus deux ? Réponds uniquement avec le nombre.' if fr else
        'Aria, What is two plus two? Answer with only the number.'))
    cursor = (await client.get('/_e2e/state')).json()['next']
    await client.post('/api/llm/process', json=body(
        'Quelle heure est-il à Zurich ?' if fr else 'What time is it in Zurich?', True))
    await events_until(client, cursor, lambda s: any(e['event'] == 'Live.Audio' for e in s['events']))
    # Identical arrival delay in enabled/disabled runs; cancellation is measured,
    # not a benchmark pause waiting for speculative work to finish.
    await asyncio.sleep(.05)
    first_text = ('Combien font douze fois sept ? Réponds uniquement avec le nombre.' if fr else
                  'What is twelve multiplied by seven? Answer with only the number.')
    queued_text = ('Combien font neuf moins quatre ? Réponds uniquement avec le nombre.' if fr else
                   'What is nine minus four? Answer with only the number.')
    first = asyncio.create_task(client.post('/api/llm/process', json=body(first_text)))
    await events_until(client, cursor, lambda s: any(
        e['event'] == 'Live.Request' and e['text'] == first_text for e in s['events']))
    submitted = time.time()
    second = asyncio.create_task(client.post('/api/llm/process', json=body(queued_text, True)))
    await events_until(client, cursor, lambda s: sum(
        e['event'] == 'ARIA.Request.Status' and e.get('status') == 'accepted'
        for e in s['events']) >= 3)
    overload = await client.post('/api/llm/process', json=body('This request must be rejected.'))
    first_result, second_result = await asyncio.gather(first, second)
    state = await events_until(client, cursor, lambda s:
        sum(e['event'] == 'Live.Reply' for e in s['events']) >= 3 and not s['speech_busy'])
    events = state['events']
    request = next(e for e in events if e['event'] == 'Live.Request' and e['text'] == queued_text)
    request_events = [e for e in events if e.get('request_id') == request['request_id']]
    accepted = next(e['time'] for e in request_events if e.get('status') == 'accepted')
    running = next(e['time'] for e in request_events if e.get('status') == 'running')
    audio = next(e['time'] for e in request_events if e['event'] == 'Live.Audio')
    flags = []
    if first_result.json().get('response', '').strip(' .') != '84': flags.append('first_answer')
    if second_result.json().get('response', '').strip(' .') != '5': flags.append('queued_answer')
    if overload.status_code != 429 or overload.json().get('error') != 'busy': flags.append('overload_not_rejected')
    if running <= accepted: flags.append('missing_queue_wait')
    if any(e['event'] == 'Live.Tool' for e in events): flags.append('unexpected_tool')
    if len([e for e in request_events if e['event'] == 'Live.Audio']) != 1: flags.append('audio_count')
    return dict(language=language, repetition=repetition, flags=flags,
                first=first_result.json(), queued=second_result.json(), overload=overload.json(),
                overload_http_status=overload.status_code,
                queue_wait_s=running-accepted, accepted_to_first_audio_s=audio-accepted,
                submitted_to_first_audio_s=audio-submitted,
                processing_to_first_audio_s=audio-running, events=events)


async def main(args):
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=args.server_run+'/aria.sock'),
                                base_url='http://localhost', timeout=120, trust_env=False) as client:
        config = (await client.get('/_live/config')).json()
        assert config['tools_mode'] == 'log_only' and config['warmup_ready']
        results = []
        for language in ('fr', 'en'):
            for repetition in range(2):
                results.append(await scenario(client, language, repetition))
                Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(language, repetition, results[-1]['flags'], flush=True)
        assert all(not row['flags'] for row in results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-run', required=True)
    parser.add_argument('--output', required=True)
    asyncio.run(main(parser.parse_args()))
