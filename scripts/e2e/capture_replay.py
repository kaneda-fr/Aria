"""Verified saved-input replay: original samples -> actual client -> Apollo.

Never opens the microphone or plays audio. Recordings stay outside source trees.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import sys
import unicodedata
import re
import wave

import numpy as np
import websockets
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.e2e.client_replay import replay_native


def load_capture(directory: Path) -> dict:
    manifest = json.loads((directory / 'capture.json').read_text())
    reference_path = directory / 'reference.json'
    if not reference_path.exists():
        raise ValueError(f'{directory}: user-confirmed reference.json required; displayed prompt is not ground truth')
    reference = json.loads(reference_path.read_text())
    text = reference.get('spoken_text')
    if reference.get('confirmed_by') != 'user' or not isinstance(text, str) or not 0 < len(text.strip()) <= 1000:
        raise ValueError('A nonempty user-confirmed spoken reference is required')
    path = directory / 'native-float32.npy'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest['sha256']['native-float32.npy'] or digest != reference.get('native_sha256'):
        raise ValueError('Capture/reference hash mismatch')
    native = np.load(path, allow_pickle=False)
    rate = manifest['native_rate']
    if (not isinstance(rate, int) or not 8000 <= rate <= 192000 or native.ndim != 1
            or native.dtype != np.float32 or not 0 < len(native) <= rate * 60
            or len(native) != manifest['frames'] or not np.isfinite(native).all()):
        raise ValueError('Expected bounded, finite mono native-rate float32 capture')
    return {'native': native, 'rate': rate, 'reference': text.strip(),
            'native_sha256': digest, 'manifest': manifest}


class WireCapture:
    """Record exactly the successfully sent PCM at client turn boundaries."""
    def __init__(self, ws, directory: Path):
        self.ws = ws
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.current = bytearray()
        self.segments = []
        self.boundaries = []

    async def send(self, data):
        await self.ws.send(data)
        if isinstance(data, str):
            message = json.loads(data)
            self.boundaries.append({'time': time.monotonic(), 'message': message})
            if message['type'] == 'utterance_start' and self.current:
                raise ValueError('New client turn before previous audio ended')
            if message['type'] in ('utterance_end', 'stop') and self.current:
                pcm = bytes(self.current)
                path = self.directory / f'segment-{len(self.segments)+1}.wav'
                with wave.open(str(path), 'wb') as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(16000)
                    audio.writeframes(pcm)
                self.segments.append({'pcm_sha256': hashlib.sha256(pcm).hexdigest(),
                                      'seconds': len(pcm)/32000, 'path': str(path)})
                self.current.clear()
        else:
            self.current.extend(data)
            if len(self.current) > 2_000_000:
                raise ValueError('Diagnostic audio exceeded the bounded capture size')


def capture_checks(reference: str, segments: list, events: list, transcripts: list) -> dict:
    def tokens(text):
        normalized = unicodedata.normalize('NFKD', text.lower())
        return re.findall(r'\w+', ''.join(c for c in normalized if not unicodedata.combining(c)))

    raw = [e for e in events if e['event'] == 'Baseline.ASR']
    sent = {s['pcm_sha256'] for s in segments}
    flags = []
    if not raw or any(e['pcm_sha256'] not in sent for e in raw):
        flags.append('asr_audio_hash_mismatch')
    if [e['text'].strip() for e in raw] != transcripts:
        flags.append('asr_to_request_mismatch')
    if tokens(' '.join(transcripts)) != tokens(reference):
        flags.append('confirmed_input_mismatch')
    return {'raw_asr': raw, 'input_trace_flags': flags, 'confirmed_input_matches': not flags}


async def exercise_captures(client, ws_url: str, artifacts: Path, report: dict):
    from baseline import characterize, settle

    for language in ('en', 'fr'):
        captured = load_capture(Path(report['capture_dir']) / language)
        # These are confirmed utterances, not assumed readings of the displayed question.
        case = {'id': 'confirmed_recording', language: captured['reference']}
        label = f'capture-{language}-confirmed_recording'
        entry = {'id': label, 'language': language, 'mode': 'capture',
                 'prompt': captured['reference'], 'native_sha256': captured['native_sha256'],
                 'input_audio_seconds': len(captured['native'])/captured['rate'],
                 'listener_notes': captured['manifest'].get('listener_notes')}
        report['cases'].append(entry)
        print(f'RUN {label}', flush=True)
        after = (await client.get('/_e2e/state')).json()['next']
        start = time.monotonic()
        try:
            async with websockets.connect(ws_url, proxy=None) as ws:
                await ws.send(json.dumps({'type': 'start', 'sample_rate': 16000, 'channels': 1,
                                         'format': 'pcm_s16le', 'chunk_ms': 40, 'client_vad': True}))
                observed = WireCapture(ws, artifacts / label)
                entry['client'] = await replay_native(observed, captured['native'], captured['rate'])
                await observed.send(json.dumps({'type': 'stop'}))
                if observed.current:
                    raise ValueError('Unfinished client audio')
                entry['wire_segments'] = observed.segments
                entry['client_boundaries'] = [{**b, 'time': b['time']-start} for b in observed.boundaries]
                output = await settle(client, after)
            entry['events'] = output['events']
            entry.update(characterize(case, language, 'capture', output['events']))
            entry.update(capture_checks(captured['reference'], observed.segments, output['events'], entry['transcripts']))
            entry['quality_flags'].extend(entry['input_trace_flags'])
            print(f"RESULT {label}: transcripts={entry['transcripts']} flags={entry['quality_flags']}", flush=True)
        except Exception as exc:
            entry['error'] = repr(exc)
            entry['pipeline_completed'] = False
            entry['events'] = (await client.get('/_e2e/state', params={'after': after})).json()['events']
            print(f'FAILED {label}: {exc}', flush=True)
        entry['scenario_seconds'] = time.monotonic()-start
        (artifacts/'baseline-progress.json').write_text(json.dumps(report, indent=2))
        import asyncio
        await asyncio.sleep(1)
