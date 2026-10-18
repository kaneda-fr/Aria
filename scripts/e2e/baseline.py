"""Matched bilingual characterization, with raw evidence and explicit quality checks.

Pipeline completion is separate from answer quality. This corpus is a small smoke
baseline, not a held-out model-selection benchmark. No device control is enabled.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import unicodedata
import wave

import httpx
import websockets
from client_replay import replay
from run import ROOT

CASES = [
    {'id': 'arithmetic', 'en': 'What is two plus two? Answer with just the number.',
     'fr': 'Combien font deux plus deux ? Réponds uniquement avec le nombre.'},
    {'id': 'capital', 'en': 'What is the capital of France? Answer in one short sentence.',
     'fr': 'Quelle est la capitale de la France ? Réponds en une courte phrase.'},
    {'id': 'story', 'en': 'Tell me a short story about a little cat.',
     'fr': 'Raconte-moi une courte histoire avec un petit chat.'},
    {'id': 'unavailable_control', 'en': 'Turn on the light in the living room.',
     'fr': 'Allume la lumière du salon.'},
]


def words(text):
    text = unicodedata.normalize('NFKD', text.lower())
    return re.findall(r'\w+', ''.join(c for c in text if not unicodedata.combining(c)))


def word_error(reference, hypothesis):
    a, b = words(reference), words(hypothesis)
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1]+1, row[j]+1, row[j-1]+(x != y)))
        row = nxt
    return {'errors': row[-1], 'reference_words': len(a), 'wer': row[-1]/max(1, len(a))}


def characterize(case, language, mode, events):
    models = [e for e in events if e['event'] == 'Baseline.Model']
    replies = [(e.get('message') or {}).get('content', '') for e in models]
    transcripts = [re.sub(r'^\[speaker=[^]]+\]\s*', '', e.get('input') or '') for e in models]
    audio = [e for e in events if e['event'] == 'E2E.AudioFetched']
    starts = [e for e in events if e['event'] == 'ARIA.LLM.Start']
    answer = ' '.join(replies)
    result = {'replies': replies, 'transcripts': transcripts, 'model_calls': len(models),
        'audio_outputs': len(audio), 'truncated': any(e.get('done_reason') == 'length' for e in models),
        'tts_languages': [e['lang'] for e in events if e['event'] == 'ARIA.TTS.Language'],
        'model_ms': [e.get('total_duration', 0)/1e6 for e in models],
        'generation_tokens': [e.get('eval_count') for e in models],
        'tools_offered': [e['tools_offered'] for e in models]}
    if mode != 'text':
        result['asr'] = word_error(case[language], ' '.join(transcripts))
    if starts and audio:
        result['first_llm_start_to_first_audio_ms'] = (audio[0]['time']-starts[0]['time'])*1000
    if models and audio:
        result['last_model_to_last_audio_ms'] = (audio[-1]['time']-models[-1]['time'])*1000
    if case['id'] == 'arithmetic':
        result['answer_check'] = bool(re.fullmatch(r'\s*(4|four|quatre)[.!]?\s*', answer, re.I))
    elif case['id'] == 'capital':
        result['answer_check'] = 'paris' in answer.lower()
    else:
        result['answer_check'] = None  # Narrative and truthful inability need explicit human-readable review.
    result['pipeline_completed'] = bool(models and audio)
    # Output keywords alone can falsely pass when the intended microphone request
    # was lost and unrelated input happened to elicit the expected answer.
    result['input_intent_check'] = None
    if case['id'] == 'capital' and mode != 'text':
        tokens = set(words(' '.join(transcripts)))
        result['input_intent_check'] = 'france' in tokens and bool(tokens & {'capital', 'capitale'})
    result['quality_flags'] = []
    if result['truncated']:
        result['quality_flags'].append('token_limit')
    if result['answer_check'] is False:
        result['quality_flags'].append('wrong_known_answer')
    if result['input_intent_check'] is False:
        result['quality_flags'].append('expected_question_not_recognized')
    if len(models) != 1:
        result['quality_flags'].append('unexpected_model_call_count')
    if len(audio) != len(models) or not audio:
        result['quality_flags'].append('missing_audio_output')
    if any(lang != language for lang in result['tts_languages']):
        result['quality_flags'].append('wrong_tts_language')
    return result


async def settle(client, after, timeout=100):
    deadline = time.monotonic() + timeout
    quiet_since = time.monotonic()
    last_count = -1
    while time.monotonic() < deadline:
        state = (await client.get('/_e2e/state', params={'after': after})).json()
        events = state['events']
        if len(events) != last_count:
            last_count, quiet_since = len(events), time.monotonic()
        models = sum(e['event'] == 'Baseline.Model' for e in events)
        starts = sum(e['event'] == 'ARIA.LLM.Start' for e in events)
        audio = sum(e['event'] == 'E2E.AudioFetched' for e in events)
        if models and models >= starts and audio and not state['speech_busy'] and time.monotonic()-quiet_since > 2:
            return state
        await asyncio.sleep(.5)
    return state  # Preserve failure evidence rather than discarding the rest of the corpus.


class ObservedSocket:
    def __init__(self, ws):
        self.ws = ws
        self.boundaries = []
        self.last_binary = None

    async def send(self, data):
        await self.ws.send(data)
        if isinstance(data, str):
            self.boundaries.append({'time': time.monotonic(), 'message': json.loads(data)})
        else:
            self.last_binary = time.monotonic()


async def exercise_baseline(base: str, artifacts: Path, microphone: bool, report: dict):
    os.environ['ARIA_CLIENT_VAD_SILENCE_MS'] = '800'
    report.update({'cases': [], 'device_actions': False, 'physical_sonos': False,
        'router': False, 'speaker_recognition': False, 'corpus': CASES,
        'profile': {'model': 'qwen2.5:3b-instruct', 'num_predict': 80, 'context': 2048,
                    'temperature': 0, 'history': 0, 'stream': False,
                    'client_vad_silence_ms': 800}})
    async with httpx.AsyncClient(base_url=base, timeout=150, trust_env=False) as client:
        deadline = time.monotonic()+120
        while True:
            try:
                ready = (await client.get('/_e2e/state', timeout=3)).json()
                break
            except (httpx.HTTPError, ValueError):
                if time.monotonic() > deadline:
                    raise TimeoutError('Baseline server not ready')
                await asyncio.sleep(1)
        assert all(ready[k] for k in ('asr', 'llm', 'tts')), ready
        assert ready['plugins'] == 0 and not ready['router'] and not ready['speaker_recognition'], ready
        report['ready'] = ready
        response = await client.post('/_e2e/baseline/start', json={'observe_asr': bool(report.get('capture_dir'))})
        response.raise_for_status()
        report['instrumentation'] = response.json()
        ws_url = base.replace('http://', 'ws://') + '/ws/asr'
        for mode in ('text', 'wav', 'mic'):
            if mode == 'mic' and not microphone:
                continue
            cases = CASES if mode != 'mic' else [CASES[1]]
            for case in cases:
                for language in ('en', 'fr'):
                    label = f"{mode}-{language}-{case['id']}"
                    entry = {'id': label, 'language': language, 'mode': mode, 'prompt': case[language]}
                    report['cases'].append(entry)
                    print(f'RUN {label}', flush=True)
                    fixture = artifacts / f"{language}-{case['id']}.wav"
                    if mode != 'text' and not fixture.exists():
                        response = await client.post('/_e2e/baseline/fixture', json={'language': language, 'text': case[language]})
                        response.raise_for_status()
                        fixture.write_bytes(response.content)
                    after = (await client.get('/_e2e/state')).json()['next']
                    t0 = time.monotonic()
                    try:
                        if mode == 'text':
                            response = await client.post('/api/llm/process', json={
                                'text': case[language], 'use_tools': False, 'speak_response': True})
                            response.raise_for_status()
                            entry['api'] = response.json()
                            output = await settle(client, after)
                        elif mode == 'wav':
                            with wave.open(str(fixture)) as audio:
                                entry['input_audio_seconds'] = audio.getnframes()/audio.getframerate()
                                pcm = audio.readframes(audio.getnframes())
                            async with websockets.connect(ws_url, proxy=None) as ws:
                                await ws.send(json.dumps({'type': 'start', 'sample_rate': 16000,
                                    'channels': 1, 'format': 'pcm_s16le', 'chunk_ms': 40, 'client_vad': True}))
                                observed = ObservedSocket(ws)
                                entry['client'] = await replay(observed, pcm)
                                entry['client_boundaries'] = [{**b, 'time': b['time']-t0} for b in observed.boundaries]
                                entry['input_delivery_seconds'] = time.monotonic()-t0
                                output = await settle(client, after)
                                await ws.send(json.dumps({'type': 'stop'}))
                        else:
                            from mac_audio import microphone_authorization
                            assert microphone_authorization() == 'authorized'
                            env = {**os.environ, 'ARIA_CLIENT_VAD': '1', 'ARIA_CLIENT_DEBUG': '1', 'ARIA_CLIENT_AGC': '0'}
                            with (artifacts / f'{label}.log').open('w') as log:
                                mic = subprocess.Popen([sys.executable, str(ROOT/'client/macos/mic_stream.py'),
                                    ws_url, '--device', 'MacBook Pro Microphone'], env=env, stdout=log, stderr=subprocess.STDOUT)
                                try:
                                    await asyncio.sleep(3)
                                    player = await asyncio.create_subprocess_exec('afplay', str(fixture))
                                    try:
                                        await asyncio.wait_for(player.wait(), 20)
                                    except asyncio.TimeoutError:
                                        player.kill()
                                        await player.wait()
                                        raise
                                    await asyncio.sleep(3)
                                finally:
                                    mic.send_signal(signal.SIGINT)
                                    try:
                                        await asyncio.to_thread(mic.wait, timeout=8)
                                    except subprocess.TimeoutExpired:
                                        mic.kill()
                                        mic.wait()
                            entry['microphone_exit_code'] = mic.returncode
                            output = await settle(client, after)
                        entry['events'] = output['events']
                        entry.update(characterize(case, language, mode, output['events']))
                        print(f"RESULT {label}: pipeline={entry['pipeline_completed']} answer={entry['answer_check']} truncated={entry['truncated']} replies={entry['replies']}", flush=True)
                    except Exception as exc:
                        entry['error'] = repr(exc)
                        entry['pipeline_completed'] = False
                        entry['events'] = (await client.get('/_e2e/state', params={'after': after})).json()['events']
                        print(f'FAILED {label}: {exc}', flush=True)
                    entry['scenario_seconds'] = time.monotonic()-t0
                    (artifacts/'baseline-progress.json').write_text(json.dumps(report, indent=2))
                    # Let the echo guard's tail expire before starting the next independent case.
                    await asyncio.sleep(1)
        if report.get('capture_dir'):
            from capture_replay import exercise_captures
            await exercise_captures(client, ws_url, artifacts, report)
    report['pipeline_cases_completed'] = sum(c['pipeline_completed'] for c in report['cases'])
    report['quality_failures'] = [c['id'] for c in report['cases'] if c.get('quality_flags') or c.get('error')]
    if report['pipeline_cases_completed'] != len(report['cases']):
        raise RuntimeError('One or more baseline cases did not complete; see case evidence')
