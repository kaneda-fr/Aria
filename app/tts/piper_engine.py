from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PiperEngine:
    cache_dir: Path
    timeout_sec: float = 10.0

    def synthesize_to_wav(self, text: str, *, voice: str, rate: float) -> Path:
        """Run Piper via subprocess and return a raw WAV path.

        ARIA expects Piper to be installed on the host (CLI `piper`).

        `voice` is treated as a model path.
        """

        if not text.strip():
            raise ValueError("Cannot synthesize empty text")
        if not voice.strip():
            raise ValueError("ARIA_TTS_VOICE must be set to a Piper model path")

        raw_dir = self.cache_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        key = _hash_key(text=text, voice=voice, rate=rate)
        out_path = raw_dir / f"{key}.wav"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        piper_bin = os.environ.get("ARIA_PIPER_BIN", "piper").strip() or "piper"
        speaker = os.environ.get("ARIA_TTS_PIPER_SPEAKER", "").strip()

        cmd: list[str] = [
            piper_bin,
            "--model",
            voice,
            "--output_file",
            str(out_path),
        ]

        # Rate control: piper supports --length_scale (inverse: <1 faster, >1 slower)
        # ARIA_TTS_RATE is defined as 1.0 normal, >1 faster, <1 slower.
        # Convert to length_scale = 1 / rate.
        if rate and rate > 0:
            length_scale = 1.0 / float(rate)
            cmd += ["--length_scale", f"{length_scale:.4f}"]

        if speaker:
            cmd += ["--speaker", speaker]

        # Piper reads text from stdin by default.
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_sec,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Piper failed (code {proc.returncode}): {stderr[-1000:]}")
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError("Piper produced no output WAV")
        return out_path


def _hash_key(*, text: str, voice: str, rate: float) -> str:
    h = hashlib.sha256()
    h.update(text.strip().encode("utf-8"))
    h.update(b"\n")
    h.update(voice.strip().encode("utf-8"))
    h.update(b"\n")
    h.update(f"{rate:.4f}".encode("ascii"))
    return h.hexdigest()[:32]
