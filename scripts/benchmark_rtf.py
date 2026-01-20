"""Benchmark real-time factor (RTF) on WAV files.

WAV files must match ARIA's strict input contract:
- PCM16LE
- 16000 Hz
- mono

Example:

```zsh
/Users/kaneda/dev/aria/.venv/bin/python scripts/benchmark_rtf.py --wav samples/*.wav
```
"""

from __future__ import annotations

import argparse
import glob
import time
import wave
from pathlib import Path

from app.asr.parakeet_onnx import ParakeetOnnx


def _read_wav_pcm16le_16k_mono(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        rate = w.getframerate()
        frames = w.getnframes()
        comptype = w.getcomptype()

        if comptype != "NONE":
            raise ValueError(f"Unsupported WAV compression {comptype} in {path}")
        if channels != 1:
            raise ValueError(f"Expected mono WAV, got {channels} channels: {path}")
        if rate != 16000:
            raise ValueError(f"Expected 16000 Hz WAV, got {rate}: {path}")
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sampwidth}: {path}")

        pcm = w.readframes(frames)
        duration_s = frames / float(rate) if rate else 0.0
        return pcm, duration_s


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ARIA Parakeet ONNX RTF on WAV inputs")
    p.add_argument("--model-path", default=None, help="Optional explicit path to model .onnx")
    p.add_argument("--wav", nargs="+", required=True, help="WAV file(s) or glob(s)")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    wav_paths: list[Path] = []
    for item in args.wav:
        matches = sorted(glob.glob(item))
        if matches:
            wav_paths.extend(Path(m) for m in matches)
        else:
            wav_paths.append(Path(item))

    wav_paths = [p for p in wav_paths if p.exists()]
    if not wav_paths:
        raise SystemExit("No WAV files found")

    asr = ParakeetOnnx(model_path=args.model_path) if args.model_path else ParakeetOnnx()

    total_audio = 0.0
    total_wall = 0.0

    for path in wav_paths:
        pcm, duration_s = _read_wav_pcm16le_16k_mono(path)
        t0 = time.perf_counter()
        text = asr.transcribe(pcm)
        wall = time.perf_counter() - t0
        rtf = (wall / duration_s) if duration_s > 0 else float("inf")

        total_audio += duration_s
        total_wall += wall

        print(
            f"{path}: audio={duration_s:.3f}s wall={wall:.3f}s rtf={rtf:.3f} text_len={len(text)}"
        )

    if total_audio > 0:
        print(f"TOTAL: audio={total_audio:.3f}s wall={total_wall:.3f}s rtf={total_wall/total_audio:.3f}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
