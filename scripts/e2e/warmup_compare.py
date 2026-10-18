"""Matched fresh-Ollama first-request comparisons on Apollo; optional Mac playback.

No microphone capture. Fixed text enters the shared request worker; measured latency
uses server request start to generated audio ready, excluding ASR/endpointing.
With --playback, also measure local request start to Mac player launch.
Each case owns a fresh isolated server/model and uses only log-only device actions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import tempfile
import time
from urllib.parse import urlsplit

import httpx

from run import audit_network

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = {'fr': 'Bonjour, quelle heure est-il?', 'en': 'Hello, what time is it?'}


def play(client, url, folder):
    path = urlsplit(url).path
    if not re.fullmatch(r'/tts/[0-9a-f]{32}\.wav', path):
        raise ValueError('Invalid audio path')
    audio = client.get(path)
    audio.raise_for_status()
    wav = folder / 'playback.wav'
    wav.write_bytes(audio.content)
    started = time.monotonic()
    subprocess.run(['/usr/bin/afplay', str(wav)], check=True)
    wav.unlink()
    return started


def wait_audio(client, after):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        response = client.get('/_e2e/state', params={'after': after})
        response.raise_for_status()
        state = response.json()
        audio = [e for e in state['events'] if e['event'] == 'Live.Audio']
        if audio:
            return state, audio[-1]
        time.sleep(.1)
    raise TimeoutError('No server audio')


def case(host, folder, remote, unit, archive, language, with_tools, playback=False):
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host]
    def remote_cmd(args, **kwargs):
        return subprocess.run(ssh + [shlex.join(args)], **kwargs)
    folder.mkdir()
    tunnel = None
    try:
        remote_cmd(['install', '-d', '-m', '700', remote + '/src'], check=True)
        with archive.open('rb') as source:
            remote_cmd(['tar', '-xf', '-', '-C', remote + '/src'], stdin=source, check=True)
        remote_cmd(['systemd-run', '--user', '--unit', unit, '--collect', '--property=RuntimeMaxSec=600',
                    'env', 'ARIA_E2E_RUN=' + remote, 'ARIA_E2E_SUITE=live',
                    'ARIA_E2E_WARMUP_TOOLS=' + str(int(with_tools)),
                    'strace', '-f', '-e', 'trace=network', '-s', '256', '-o', remote + '/network.trace',
                    'unshare', '--user', '--map-root-user', '--net',
                    'bash', remote + '/src/scripts/e2e/launch.sh'], check=True)
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        with (folder / 'tunnel.log').open('w') as log:
            tunnel = subprocess.Popen(ssh[:-1] + ['-N', '-o', 'ExitOnForwardFailure=yes',
                '-L', f'127.0.0.1:{port}:{remote}/aria.sock', host], stdout=log, stderr=log)
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=180, trust_env=False) as client:
            deadline = time.monotonic() + 180
            while True:
                try:
                    config = client.get('/_live/config', timeout=2).json()
                    if config.get('warmup_ready'):
                        break
                except (httpx.HTTPError, ValueError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Startup/warmup did not complete')
                time.sleep(1)
            assert config['tools_mode'] == 'log_only'
            assert config['speaker_ready'] and config['warmup']['tools'] == with_tools
            cursor = client.get('/_e2e/state').json()['next']
            ready = client.post('/_live/ready').json()
            assert ready['spoken']
            state, audio = wait_audio(client, cursor)
            if playback:
                play(client, audio['url'], folder)
            cursor = state['next']
            started = time.monotonic()
            response = client.post('/api/llm/process', json={
                'text': QUESTIONS[language], 'session_id': 'first-request',
                'speaker_user': 'unknown', 'speak_response': True, 'use_tools': True})
            response.raise_for_status()
            reply = response.json()
            assert reply['success'] and reply['spoken'], reply
            state, audio = wait_audio(client, cursor)
            player_started = play(client, audio['url'], folder) if playback else None
            models = [e for e in state['events'] if e['event'] == 'Live.Model']
            requests = [e for e in state['events'] if e['event'] == 'Live.Request']
            assert len(requests) == 1 and models and all(e['tools_offered'] for e in models)
            assert models[0]['history_messages'] == 0
            assert audio['request_id'] == requests[0]['request_id']
            result = {'language': language, 'tools_warmup': with_tools, 'config': config,
                      'mac_playback': playback,
                      'request_to_audio_ready_s': audio['time'] - requests[0]['time'],
                      'request_to_player_launch_s': player_started - started if playback else None,
                      'reply': reply, 'models': models, 'events': state['events'],
                      'warmup_events': client.get('/_e2e/state').json()['events'][:cursor]}
            assert not any(e['event'] in ('Live.Tool', 'Live.Command', 'Live.Request')
                           for e in result['warmup_events'])
            result['model_s'] = sum(e['total_duration'] for e in models) / 1e9
            result['prompt_s'] = sum(e['prompt_eval_duration'] for e in models) / 1e9
            result['generation_s'] = sum(e['eval_duration'] for e in models) / 1e9
            (folder / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if tunnel:
            tunnel.terminate()
            tunnel.wait(timeout=10)
        remote_cmd(['systemctl', '--user', 'stop', unit], capture_output=True)
        for name in ('server.log', 'conversation.jsonl', 'ollama.log', 'network.trace', 'routes.txt'):
            output = remote_cmd(['cat', remote + '/' + name], capture_output=True)
            if output.returncode == 0:
                (folder / name).write_bytes(output.stdout)
    audit = audit_network((folder / 'network.trace').read_text())
    (folder / 'network-audit.json').write_text(json.dumps(audit, indent=2))
    assert audit['passed'], audit
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='apollo')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--playback', action='store_true', help='Opt in to Mac audio output; never captures microphone')
    args = parser.parse_args()
    if not re.fullmatch(r'[\w.@-]+', args.host) or args.host.startswith('-'):
        parser.error('Invalid SSH host')
    if not 1 <= args.repeats <= 3:
        parser.error('repeats must be 1..3')
    os.umask(0o077)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-warmup-' + os.urandom(3).hex()
    folder = Path.home() / 'Library/Logs/Aria/e2e' / run_id
    folder.mkdir(parents=True)
    print('COMPARISON', folder, flush=True)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / 'source.tar'
        subprocess.run(['tar', '--no-xattrs', '--exclude=__pycache__', '--exclude=*.pyc',
                        '-cf', str(archive), 'app', 'scripts/e2e'], cwd=ROOT, check=True)
        (folder / 'source.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
        order = [('fr', False), ('fr', True), ('en', True), ('en', False)]
        for repeat in range(args.repeats):
            for language, enabled in (order if repeat % 2 == 0 else reversed(order)):
                name = f'{len(results)+1}-{language}-{int(enabled)}'
                remote = f'/data/aria/e2e/runs/{run_id}-{name}'
                result = case(args.host, folder / name, remote,
                              f'aria-warmup-{run_id.lower()}-{name}', archive, language, enabled, args.playback)
                results.append(result)
                (folder / 'report.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
                print(json.dumps({k: result[k] for k in ('language', 'tools_warmup',
                    'request_to_audio_ready_s', 'request_to_player_launch_s',
                    'model_s', 'prompt_s', 'generation_s')}), flush=True)


if __name__ == '__main__':
    main()
