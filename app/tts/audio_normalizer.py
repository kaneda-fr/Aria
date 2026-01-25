from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SonosWavSpec:
    sample_rate: int = 44100
    channels: int = 2
    sample_fmt: str = "s16"


def normalize_for_sonos(in_path: Path, out_path: Path, *, volume: float) -> Path:
    """Convert WAV to Sonos-friendly WAV via ffmpeg.

    Requirements:
    - PCM s16le
    - stereo
    - 44100 Hz
    - gain scaling via volume filter
    """

    if volume <= 0:
        volume = 0.01

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = os.environ.get("ARIA_FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(in_path),
        "-ac",
        "2",
        "-ar",
        "44100",
        "-sample_fmt",
        "s16",
        "-filter:a",
        f"volume={volume}",
        str(out_path),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg normalize failed (code {proc.returncode}): {stderr[-1000:]}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced no output WAV")
    return out_path
