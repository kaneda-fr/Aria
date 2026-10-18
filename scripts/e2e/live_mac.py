"""Live Apollo inference with Mac microphone and Mac playback; Ctrl+C stops both.

Uses the existing isolated Apollo conversation profile. No Sonos or home actions.
All ASR/LLM/Piper processing stays on Apollo; Mac only captures and plays audio.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime, timezone
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'client/macos'))
import mic_stream as mic
from run import audit_network


async def conversation(base: str, artifacts: Path, device: str, audio_url: str | None = None) -> None:
    muted = threading.Event()
    first_frame = threading.Event()
    played = asyncio.Event()
    timings = artifacts / 'timings.jsonl'
    last_turn = None
    last_speech = None
    turn_speech_end = None
    pending_turns = deque(maxlen=32)
    request_turns = {}

    def record(event, **fields):
        with timings.open('a') as output:
            output.write(json.dumps({'event': event, 'monotonic': time.monotonic(), **fields}) + '\n')

    async with httpx.AsyncClient(base_url=base, timeout=150) as client:
        deadline = time.monotonic() + 150
        while True:
            try:
                state = (await client.get('/_e2e/state', timeout=3)).json()
                if all(state[k] for k in ('asr', 'llm', 'tts')):
                    break
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Apollo did not become ready; inspect server.log')
            await asyncio.sleep(1)
        config = (await client.get('/_live/config')).json()
        assert config['tools_mode'] == 'log_only', 'Live tools must be log-only'
        assert config['warmup_ready'] and config['warmup']['tools'], 'Tool warm-up must finish before capture'
        if config['history_max_messages'] != 12 or not config['speaker_ready']:
            raise RuntimeError(f'Live history/speaker configuration not ready: {config}')
        (artifacts / 'config.json').write_text(json.dumps(config, indent=2))
        cursor = state['next']
        print('Apollo ready: history = 6 exchanges; speaker attribution active; tools LOG ONLY; exact logging ON.', flush=True)

        async def playback():
            nonlocal cursor
            while True:
                response = await client.get('/_e2e/state', params={'after': cursor}, timeout=10)
                response.raise_for_status()
                current = response.json()
                cursor = current['next']
                for event in current['events']:
                    if event['event'].startswith('Live.') or event['event'] == 'ARIA.SPK':
                        with (artifacts / 'conversation.jsonl').open('a', encoding='utf-8') as output:
                            output.write(json.dumps(event, ensure_ascii=False) + '\n')
                        if event['event'] == 'Live.Request':
                            if event['source'] == 'voice':
                                request_turns[event['request_id']] = pending_turns[0] if len(pending_turns) == 1 else None
                                pending_turns.clear()
                                if len(request_turns) > 32:
                                    request_turns.pop(next(iter(request_turns)))
                            label = 'YOU' if event['source'] == 'voice' else 'SETUP/API'
                            print(f"{label} [{event['speaker_user']} / {event['speaker_status']}]: {event['text']}", flush=True)
                        elif event['event'] == 'Live.Reply':
                            print('ARIA:', event['text'], flush=True)
                        elif event['event'] == 'ARIA.SPK':
                            print('SPEAKER:', json.dumps(event, ensure_ascii=False), flush=True)
                        elif event['event'] in ('Live.Tool', 'Live.Command'):
                            print('LOG ONLY:', json.dumps(event, ensure_ascii=False), flush=True)
                    if event['event'] in ('ARIA.ASR.Error', 'ARIA.LLM.Error', 'ARIA.TTS.Error'):
                        print('ARIA error:', event, flush=True)
                    if event['event'] != 'Live.Audio':
                        continue
                    path = urlsplit(event['url']).path
                    if not re.fullmatch(r'/tts/[0-9a-f]{32}\.wav', path):
                        raise ValueError('Invalid Apollo audio path')
                    muted.set()
                    proc = None
                    try:
                        audio = await client.get(path)
                        audio.raise_for_status()
                        if len(audio.content) > 10 * 1024 * 1024:
                            raise ValueError('Audio exceeds playback bound')
                        with tempfile.TemporaryDirectory(prefix='aria-play-') as tmp:
                            wav = Path(tmp) / 'reply.wav'
                            wav.write_bytes(audio.content)
                            turn = request_turns.pop(event['request_id'], None)
                            elapsed = time.monotonic() - turn[0] if turn else None
                            record('mac_playback_start', path=path, bytes=len(audio.content),
                                   request_id=event['request_id'],
                                   client_turn_end_to_player_start_s=elapsed,
                                   last_vad_speech_to_player_start_s=(time.monotonic() - turn[1]
                                                                       if turn and turn[1] else None),
                                   timing_association='single_pending_turn' if turn else 'unavailable_or_ambiguous')
                            print('ARIA speaking on Mac' + (f' — {elapsed:.2f}s after turn end' if elapsed else ''), flush=True)
                            proc = await asyncio.create_subprocess_exec('/usr/bin/afplay', str(wav))
                            if await proc.wait() != 0:
                                raise RuntimeError('Mac audio playback failed')
                            record('mac_playback_complete', path=path, request_id=event['request_id'])
                        await asyncio.sleep(.3)
                    finally:
                        if proc and proc.returncode is None:
                            proc.terminate()
                            await proc.wait()
                        muted.clear()
                    played.set()
                    if first_frame.is_set():
                        print('READY — speak in French or English. Ctrl+C stops Aria.', flush=True)
                await asyncio.sleep(.1)

        player = asyncio.create_task(playback())
        original_stream, original_send = mic.sd.InputStream, mic._send_msg
        original_vad = mic.webrtcvad.Vad

        def observed_send(payload):
            nonlocal last_turn, turn_speech_end
            if payload.get('type') == 'utterance_end':
                last_turn = time.monotonic()
                turn_speech_end = last_speech
                pending_turns.append((last_turn, turn_speech_end))
                record('client_turn_end', last_vad_speech=last_speech)
                print('Thinking on Apollo...', flush=True)
            return original_send(payload)

        class ObservedVad:
            def __init__(self, *args, **kwargs):
                self.inner = original_vad(*args, **kwargs)

            def is_speech(self, frame, rate):
                nonlocal last_speech
                result = self.inner.is_speech(frame, rate)
                if result:
                    last_speech = time.monotonic()
                return result

        def input_stream(**kwargs):
            callback = kwargs['callback']
            def capture(indata, frames, timing, status):
                if not first_frame.is_set():
                    first_frame.set()
                    print('READY — speak in French or English. Ctrl+C stops Aria.', flush=True)
                # Continue sending silence through the normal VAD/resampler while
                # playback is active; do not feed the assistant's voice back in.
                callback(np.zeros_like(indata) if muted.is_set() else indata, frames, timing, status)
            return original_stream(**{**kwargs, 'callback': capture})

        mic_task = None
        try:
            response = await client.post('/_live/ready')
            response.raise_for_status()
            result = response.json()
            if not result.get('success') or not result.get('spoken'):
                raise RuntimeError(result)
            print('ARIA:', result['response'], flush=True)
            ready_audio = asyncio.create_task(played.wait())
            done, _ = await asyncio.wait([ready_audio, player], timeout=90, return_when=asyncio.FIRST_COMPLETED)
            if player in done:
                await player
            if not played.is_set():
                ready_audio.cancel()
                raise TimeoutError('No Mac playback confirmation')
            mic.sd.InputStream, mic._send_msg = input_stream, observed_send
            mic.webrtcvad.Vad = ObservedVad
            mic_task = asyncio.create_task(mic.run_client(audio_url or base.replace('http://', 'ws://') + '/ws/asr',
                mic.AriaAudioContract(), device=device))
            done, _ = await asyncio.wait([mic_task, player], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
        finally:
            for task in (mic_task, player):
                if task:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            mic.sd.InputStream, mic._send_msg = original_stream, original_send
            mic.webrtcvad.Vad = original_vad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='apollo')
    parser.add_argument('--device', default='MacBook Pro Microphone')
    parser.add_argument('--audio-url', help='Optional LAN WebSocket ingress for reconnect tests; playback keeps its SSH tunnel')
    args = parser.parse_args()
    if not re.fullmatch(r'[\w.@-]+', args.host) or args.host.startswith('-'):
        parser.error('Expected SSH alias or user@host')
    os.umask(0o077)
    for key, value in {'ARIA_CLIENT_VAD': '1', 'ARIA_CLIENT_AGC': '0',
                       'ARIA_CLIENT_VAD_AGGR': '3', 'ARIA_CLIENT_VAD_START_FRAMES': '3',
                       'ARIA_CLIENT_VAD_SNR_DB': '8', 'ARIA_CLIENT_VAD_MIN_RMS': '0.005',
                       'ARIA_CLIENT_VAD_SILENCE_MS': '800'}.items():
        os.environ.setdefault(key, value)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-live-' + os.urandom(3).hex()
    remote = '/data/aria/e2e/runs/' + run_id
    unit = 'aria-live-' + run_id.lower()
    artifacts = Path.home() / 'Library/Logs/Aria/e2e' / run_id
    artifacts.mkdir(parents=True, mode=0o700)
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', args.host]
    tunnel = None
    def remote_command(argv, **kwargs):
        return subprocess.run(ssh + [shlex.join(argv)], **kwargs)
    print(f'All inference runs on Apollo. Mac handles microphone and playback.\nLogs: {artifacts}', flush=True)
    metadata = {'run_id': run_id, 'remote': remote, 'unit': unit, 'pid': os.getpid()}
    (artifacts / 'live.json').write_text(json.dumps(metadata, indent=2))
    try:
        remote_command(['install', '-d', '-m', '700', remote + '/src'], check=True)
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'source.tar'
            subprocess.run(['tar', '--no-xattrs', '--exclude=__pycache__', '--exclude=*.pyc',
                            '--exclude=*.backup', '-cf', str(archive), 'app', 'scripts/e2e'], cwd=ROOT, check=True)
            metadata['source_archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
            with archive.open('rb') as source:
                remote_command(['tar', '-xf', '-', '-C', remote + '/src'], stdin=source, check=True)
        remote_command(['systemd-run', '--user', '--unit', unit, '--collect', '--property=RuntimeMaxSec=10800',
            'env', 'ARIA_E2E_RUN=' + remote, 'ARIA_E2E_SUITE=live', 'strace', '-f', '-e', 'trace=network', '-s', '256',
            '-o', remote + '/network.trace', 'unshare', '--user', '--map-root-user', '--net',
            'bash', remote + '/src/scripts/e2e/launch.sh'], check=True)
        # An installed monitor/ingress follows the active development instance.
        # Atomic replacement prevents readers observing a partial pointer.
        pointer_code = ('from pathlib import Path\nimport json\n'
                        'p=Path("/data/aria/monitor/active.json")\n'
                        'if p.parent.is_dir():\n'
                        ' t=p.with_suffix(".tmp")\n'
                        ' t.write_text(json.dumps(' + repr({'run_dir': remote, 'tools_mode': 'log_only'}) + '))\n'
                        ' t.replace(p)\n')
        remote_command(['/data/aria/dev/.venv/bin/python', '-c', pointer_code], check=True)
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        metadata['port'] = port
        (artifacts / 'live.json').write_text(json.dumps(metadata, indent=2))
        with (artifacts / 'tunnel.log').open('w') as log:
            tunnel = subprocess.Popen(ssh[:-1] + ['-N', '-o', 'ExitOnForwardFailure=yes', '-L',
                f'127.0.0.1:{port}:{remote}/aria.sock', args.host], stdout=log, stderr=log)
        asyncio.run(conversation(f'http://127.0.0.1:{port}', artifacts, args.device, args.audio_url))
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel:
            tunnel.terminate()
            tunnel.wait(timeout=10)
        remote_command(['systemctl', '--user', 'stop', unit], capture_output=True)
        for name in ('server.log', 'ollama.log', 'network.trace', 'routes.txt'):
            result = remote_command(['cat', remote + '/' + name], capture_output=True)
            if result.returncode == 0:
                (artifacts / name).write_bytes(result.stdout)
        result = remote_command(['cat', remote + '/conversation.jsonl'], capture_output=True)
        if result.returncode == 0:
            (artifacts / 'conversation-apollo.jsonl').write_bytes(result.stdout)
        trace = artifacts / 'network.trace'
        if trace.exists():
            (artifacts / 'network-audit.json').write_text(json.dumps(audit_network(trace.read_text()), indent=2))
        print('Aria stopped. Apollo test service and SSH tunnel closed.', flush=True)


if __name__ == '__main__':
    main()
