"""Drive the unmodified Mac client's sender with synthetic native-rate audio."""
import asyncio
import contextlib
import os
from pathlib import Path
import sys

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from client.macos.mic_stream import AriaAudioContract, ClientStats, _send_loop  # noqa: E402


async def replay(ws, pcm16: bytes) -> dict:
    # Match the inspected microphone's native 96 kHz input while exercising the
    # real client's 16 kHz resampler, VAD, framing and utterance-boundary messages.
    rate = 96000
    samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768
    native = soxr.resample(samples, 16000, rate).astype(np.float32)
    return await replay_native(ws, native, rate)


async def replay_native(ws, native: np.ndarray, rate: int) -> dict:
    """Feed original microphone floats through the real sender, without capture."""
    native = np.concatenate((np.zeros(rate // 2, dtype=np.float32), native,
                             np.zeros(rate * 2, dtype=np.float32)))
    queue = asyncio.Queue(maxsize=50)
    stop = asyncio.Event()
    stats = ClientStats()
    os.environ["ARIA_CLIENT_AGC"] = "0"
    sender = asyncio.create_task(_send_loop(
        ws, queue, stop, in_sample_rate=rate, contract=AriaAudioContract(),
        stats=stats, client_vad_enabled=True,
    ))
    try:
        block_frames = max(1, int(round(rate * .04)))
        for offset in range(0, len(native), block_frames):
            if sender.done():
                raise RuntimeError("Client sender exited before fixture was delivered")
            await asyncio.wait_for(queue.put(native[offset:offset+block_frames]), timeout=2)
            stats.enqueued_blocks += 1
            await asyncio.sleep(0.04)
        deadline = asyncio.get_running_loop().time() + 5
        while not queue.empty():
            if asyncio.get_running_loop().time() > deadline or sender.done():
                raise RuntimeError("Client sender did not drain fixture")
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.2)
    finally:
        stop.set()
        try:
            await asyncio.wait_for(sender, timeout=2)
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
    return {"native_rate": rate, "blocks": stats.enqueued_blocks,
            "client_vad": True, "physical_capture": False}
