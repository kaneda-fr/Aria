"""Play new Apollo replies on macOS; never opens a microphone or starts inference."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time
from urllib.parse import urlsplit

import httpx


async def listen(url, artifacts):
    async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=15) as client:
        for attempt in range(40):
            try:
                config = (await client.get('/_live/config')).raise_for_status().json()
                break
            except httpx.TransportError:
                if attempt == 39:
                    raise
                await asyncio.sleep(.25)
        if config.get('tools_mode') != 'log_only' or not config.get('warmup_ready'):
            raise RuntimeError('Expected ready, log-only Apollo instance')
        # Start at the current cursor: never replay earlier benchmark or conversation audio.
        cursor = (await client.get('/_e2e/state')).raise_for_status().json()['next']
        (artifacts / 'ready.json').write_text(json.dumps({
            'pid': os.getpid(), 'url': url, 'mac_capture': False,
            'mac_playback': True, 'cursor': cursor, 'config': config}))
        print('READY — Mac playback only; microphone OFF.', flush=True)
        while True:
            state = (await client.get('/_e2e/state', params={'after': cursor})).raise_for_status().json()
            cursor = state['next']
            for event in state['events']:
                if event['event'] != 'Live.Audio':
                    continue
                path = urlsplit(event['url']).path
                if not re.fullmatch(r'/tts/[0-9a-f]{32}\.wav', path):
                    raise ValueError('Invalid Apollo audio path')
                with tempfile.TemporaryDirectory(prefix='aria-playback-') as temp:
                    wav = Path(temp) / 'reply.wav'
                    size = 0
                    async with client.stream('GET', path) as response:
                        response.raise_for_status()
                        with wav.open('wb') as output:
                            async for chunk in response.aiter_bytes():
                                size += len(chunk)
                                if size > 10 * 1024 * 1024:
                                    raise ValueError('Audio exceeds playback bound')
                                output.write(chunk)
                    started = time.time()
                    proc = await asyncio.create_subprocess_exec('/usr/bin/afplay', str(wav))
                    try:
                        code = await proc.wait()
                    finally:
                        if proc.returncode is None:
                            proc.terminate()
                            await proc.wait()
                    row = {'request_id': event.get('request_id'), 'audio_ready': event['time'],
                           'player_started': started, 'player_finished': time.time(),
                           'exit_code': code, 'bytes': size}
                    with (artifacts / 'playback.jsonl').open('a') as log:
                        log.write(json.dumps(row) + '\n')
                    print(json.dumps(row), flush=True)
                    if code:
                        raise RuntimeError('Mac audio playback failed')
            await asyncio.sleep(.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='apollo')
    parser.add_argument('--run', required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[\w.@-]+', args.host) or args.host.startswith('-'):
        parser.error('Invalid SSH host')
    if not re.fullmatch(r'/data/aria/e2e/runs/[\w-]+', args.run):
        parser.error('Expected isolated Apollo run path')
    os.umask(0o077)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / 'ready.json').unlink(missing_ok=True)
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    with (args.artifacts / 'tunnel.log').open('a') as log:
        tunnel = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
            '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=15',
            '-o', 'ServerAliveCountMax=3', '-N', '-L',
            f'127.0.0.1:{port}:{args.run}/aria.sock', args.host], stdout=log, stderr=log)
        try:
            asyncio.run(listen(f'http://127.0.0.1:{port}', args.artifacts))
        except KeyboardInterrupt:
            pass
        finally:
            tunnel.terminate()
            tunnel.wait(timeout=10)
            (args.artifacts / 'ready.json').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
