"""User-controlled, local microphone capture and replay for acoustic diagnosis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import subprocess
import sys
import threading
import time
import wave

import numpy as np
import sounddevice as sd
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from client.macos.mic_stream import _select_input_device


PROMPTS = {
    'en': 'What is the capital of France? Answer in one short sentence.',
    'fr': 'Quelle est la capitale de la France ? Réponds en une courte phrase.',
}


def enter_pressed() -> bool:
    if select.select([sys.stdin], [], [], .1)[0]:
        if sys.stdin.readline() == '':
            raise EOFError('Terminal input closed')
        return True
    return False


def capture(device: str, max_seconds: int, stop_requested=enter_pressed, announce=print):
    """Bounded native-rate float capture; no playback, VAD, gain, or network IO."""
    selected = _select_input_device(device)
    info = sd.query_devices(selected, 'input')
    rate = int(round(info['default_samplerate']))
    data = np.empty(rate * max_seconds, dtype=np.float32)
    ready = threading.Event()
    full = threading.Event()
    count = 0
    status_counts = {}

    def callback(indata, frames, timing, status):
        nonlocal count
        if status:
            key = str(status)
            status_counts[key] = status_counts.get(key, 0) + 1
        n = min(frames, len(data) - count)
        if n:
            data[count:count+n] = indata[:n, 0]
            count += n
            ready.set()
        if count == len(data):
            full.set()

    announce(f"Opening {info['name']} at {rate} Hz…", flush=True)
    start = time.monotonic()
    stop_reason = 'enter'
    with sd.InputStream(device=selected, samplerate=rate, channels=1,
                        dtype='float32', blocksize=max(1, rate // 25), callback=callback):
        if not ready.wait(5):
            raise RuntimeError('No microphone frames received; check macOS microphone permission.')
        ready_offset = count
        announce('READY — speak now. Leave two seconds of silence after the sentence, then press ENTER to stop.', flush=True)
        try:
            while not full.is_set():
                if stop_requested():
                    break
                if time.monotonic() - start > max_seconds + 5:
                    stop_reason = 'wall_timeout'
                    break
            else:
                stop_reason = 'duration_limit'
        except KeyboardInterrupt:
            stop_reason = 'ctrl_c'
    # The stream is stopped/closed before returning any samples to replay code.
    announce('STOPPED — microphone closed.', flush=True)
    metadata = {'device': info['name'], 'native_rate': rate, 'frames': count,
                'seconds': count / rate, 'ready_sample_offset': ready_offset,
                'stop_reason': stop_reason, 'capture_status': status_counts,
                'peak': float(np.max(np.abs(data[:count]))),
                'rms': float(np.sqrt(np.mean(data[:count] ** 2)))}
    return data[:count].copy(), rate, metadata


def save_audio(directory: Path, samples, rate: int) -> dict:
    """Preserve original floats plus listenable PCM and a 16 kHz analysis copy."""
    np.save(directory / 'native-float32.npy', samples)
    analysis = soxr.resample(samples, rate, 16000, quality='HQ')
    for filename, audio, sample_rate in [('native.wav', samples, rate), ('analysis-16k.wav', analysis, 16000)]:
        pcm = (np.clip(audio, -1, 1) * 32767).astype('<i2')
        with wave.open(str(directory / filename), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(pcm.tobytes())
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ['native-float32.npy', 'native.wav', 'analysis-16k.wav']}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=['both', 'en', 'fr'], default='both')
    parser.add_argument('--device', default='MacBook Pro Microphone')
    parser.add_argument('--max-seconds', type=int, default=20)
    args = parser.parse_args()
    if not 3 <= args.max_seconds <= 60:
        parser.error('--max-seconds must be 3..60')
    if not sys.stdin.isatty():
        parser.error('Run this command in an interactive Terminal.')
    os.umask(0o077)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + secrets.token_hex(3)
    root = Path.home() / 'Library/Logs/Aria/captures' / run_id
    root.mkdir(parents=True, mode=0o700)
    print(f'Local recordings: {root}', flush=True)
    print('Keep other audio sources quiet. Nothing is sent to Apollo during capture.', flush=True)
    languages = ['en', 'fr'] if args.language == 'both' else [args.language]
    for language in languages:
        print(f'\n{language.upper()} — say only this after READY:\n{PROMPTS[language]}', flush=True)
        input('Press ENTER to arm the microphone (Ctrl+C here cancels): ')
        samples, rate, metadata = capture(args.device, args.max_seconds)
        directory = root / language
        directory.mkdir(mode=0o700)
        metadata.update({'language': language, 'expected_text': PROMPTS[language],
                         'captured_utc': datetime.now(timezone.utc).isoformat(),
                         'sha256': save_audio(directory, samples, rate),
                         'analysis_resampling': 'soxr HQ offline; native floats retained for exact client replay'})
        manifest = directory / 'capture.json'
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Saved {metadata["seconds"]:.2f} seconds. REPLAY — listen to your recording now.', flush=True)
        if metadata['capture_status']:
            print(f'Capture warnings: {metadata["capture_status"]}', flush=True)
        if metadata['stop_reason'] != 'enter':
            print(f'Recording ended by {metadata["stop_reason"]}.', flush=True)
        replay = subprocess.run(['afplay', str(directory / 'native.wav')], timeout=args.max_seconds + 10)
        metadata['replay_exit_code'] = replay.returncode
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        metadata['listener_notes'] = input('Replay finished. What did you hear? (clean / extra words / other notes): ')
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        exact = input('Did you say EXACTLY the displayed sentence? [y/N]: ').strip().lower()
        spoken = PROMPTS[language] if exact in ('y', 'yes') else input('Type the words you actually said (blank = unconfirmed): ').strip()
        if spoken:
            (directory / 'reference.json').write_text(json.dumps({
                'confirmed_by': 'user', 'spoken_text': spoken,
                'native_sha256': metadata['sha256']['native-float32.npy'],
            }, indent=2) + '\n')
        else:
            print('Reference unconfirmed; automated capture evaluation will require confirmation.', flush=True)
    print(f'\nDONE — recordings and notes saved to:\n{root}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print('\nStopped by user. Any completed recordings remain saved.', flush=True)
