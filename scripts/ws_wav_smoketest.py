from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import wave
from pathlib import Path

import websockets


def _read_pcm16le_16k_mono(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getcomptype() != "NONE":
            raise SystemExit(f"Unsupported WAV compression {w.getcomptype()} in {path}")
        if w.getnchannels() != 1:
            raise SystemExit(f"Expected mono WAV, got {w.getnchannels()} channels in {path}")
        if w.getframerate() != 16000:
            raise SystemExit(f"Expected 16000 Hz WAV, got {w.getframerate()} Hz in {path}")
        if w.getsampwidth() != 2:
            raise SystemExit(f"Expected 16-bit PCM WAV, got sample width {w.getsampwidth()} in {path}")
        return w.readframes(w.getnframes())


def _chunk_bytes(pcm: bytes, *, chunk_ms: int) -> list[bytes]:
    bytes_per_ms = 16000 * 2 // 1000  # 16kHz * 2 bytes/sample
    chunk_size = bytes_per_ms * chunk_ms
    if chunk_size <= 0:
        raise SystemExit("Invalid chunk_ms")
    out: list[bytes] = []
    for off in range(0, len(pcm) - (len(pcm) % chunk_size), chunk_size):
        out.append(pcm[off : off + chunk_size])
    return out


async def main() -> int:
    p = argparse.ArgumentParser(description="ARIA WebSocket WAV smoke test")
    p.add_argument("url", help="ws://host:port/ws/asr")
    p.add_argument("--wav", default="test.wav", help="Path to 16kHz mono PCM16 WAV")
    p.add_argument("--chunk-ms", type=int, default=40, help="Chunk size to send (20-60)")
    args = p.parse_args()

    wav_path = Path(args.wav).resolve()
    pcm = _read_pcm16le_16k_mono(wav_path)
    chunks = _chunk_bytes(pcm, chunk_ms=args.chunk_ms)

    # Add some trailing silence so the server can finalize the utterance if it wants.
    silence_ms = 1000
    silence_bytes = b"\x00\x00" * int(16000 * silence_ms / 1000)
    chunks.extend(_chunk_bytes(silence_bytes, chunk_ms=args.chunk_ms))

    start_msg = {
        "type": "start",
        "sample_rate": 16000,
        "channels": 1,
        "format": "pcm_s16le",
        "chunk_ms": int(args.chunk_ms),
    }

    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps(start_msg))

        # Spec change: server prints transcripts and does not send them back.
        async def drain() -> None:
            async for _msg in ws:
                continue

        recv_task = asyncio.create_task(drain())

        for ch in chunks:
            await ws.send(ch)

        try:
            await ws.send(json.dumps({"type": "stop"}))
        except Exception:
            pass

        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await recv_task

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
