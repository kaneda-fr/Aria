from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import json
import os
import signal
import sys
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import numpy as np
import sounddevice as sd
import soxr
import websockets

try:
    import webrtcvad
except Exception:  # pragma: no cover
    webrtcvad = None


@dataclass(frozen=True, slots=True)
class AriaAudioContract:
    sample_rate: int = 16000
    channels: int = 1
    fmt: str = "pcm_s16le"
    chunk_ms: int = 40

    @property
    def frames_per_chunk(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000)

    @property
    def bytes_per_chunk(self) -> int:
        return self.frames_per_chunk * 2


@dataclass(slots=True)
class ClientStats:
    enqueued_blocks: int = 0
    dropped_queue_blocks: int = 0


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _send_msg(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _normalize_ws_url(raw: str) -> str:
    """Normalize user-provided URL into a ws(s)://.../ws/asr endpoint."""

    raw = raw.strip()
    parsed = urlparse(raw)

    # Allow passing host:port without scheme.
    if parsed.scheme == "" and parsed.netloc == "":
        parsed = urlparse("ws://" + raw)
    # urlparse("localhost:8000") treats "localhost" as a scheme; correct that.
    if "://" not in raw and parsed.scheme and parsed.netloc == "":
        parsed = urlparse("ws://" + raw)

    scheme = parsed.scheme
    if scheme in {"http", "https"}:
        scheme = "wss" if scheme == "https" else "ws"
    if scheme not in {"ws", "wss"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

    netloc = parsed.netloc
    path = parsed.path or ""
    if not path.endswith("/ws/asr"):
        # If user provided base URL, append endpoint.
        if path.endswith("/"):
            path = path[:-1]
        if path == "":
            path = "/ws/asr"
        else:
            path = path + "/ws/asr"

    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


def _select_input_device(spec: str | None) -> int | None:
    """Return a sounddevice input device index.

    - None: use system default input
    - int string: use that PortAudio device index
    - otherwise: case-insensitive substring match on device name
    """

    if spec is None or spec.strip() == "":
        return None

    spec = spec.strip()
    try:
        idx = int(spec)
    except ValueError:
        idx = None

    if idx is not None:
        info = sd.query_devices(idx, "input")
        if int(info.get("max_input_channels", 0)) <= 0:
            raise ValueError(f"Device {idx} is not an input device")
        return idx

    # Name substring
    devices = sd.query_devices()
    matches: list[int] = []
    for i, d in enumerate(devices):
        try:
            if int(d.get("max_input_channels", 0)) <= 0:
                continue
        except Exception:
            continue
        name = str(d.get("name", ""))
        if spec.lower() in name.lower():
            matches.append(i)

    if not matches:
        raise ValueError(f"No input device matches {spec!r}")
    if len(matches) > 1:
        names = [str(sd.query_devices(i, "input").get("name", "")) for i in matches[:10]]
        raise ValueError(f"Multiple devices match {spec!r}: {names}")
    return matches[0]


async def _recv_loop(ws: Any, stop_event: asyncio.Event) -> None:
    """Drain incoming messages and detect server close.

    Spec change: the server prints transcripts and does not send them back.
    We still keep a receive loop so close frames are processed promptly.
    """

    try:
        async for _message in ws:
            if stop_event.is_set():
                return
            # Ignore any unexpected messages.
            continue
    except asyncio.CancelledError:
        raise
    except Exception:
        stop_event.set()
        return


async def _send_loop(
    ws: Any,
    queue: asyncio.Queue[np.ndarray],
    stop_event: asyncio.Event,
    *,
    in_sample_rate: int,
    contract: AriaAudioContract,
    stats: ClientStats,
    client_vad_enabled: bool,
) -> None:
    # IMPORTANT CONTEXT:
    # - We capture at the device's native rate to avoid drift/timing artifacts seen
    #   when forcing rates at capture on some macOS setups.
    # - We resample client-side (device rate -> 16k) so the server can rely on a
    #   stable, uniform audio contract.
    # - We explicitly avoid ffmpeg/avfoundation due to prior timing issues.
    resampler = soxr.ResampleStream(
        float(in_sample_rate),
        float(contract.sample_rate),
        int(contract.channels),
        dtype="float32",
        quality="HQ",
    )
    out_bytes = bytearray()
    debug = os.environ.get("ARIA_CLIENT_DEBUG", "").strip() not in {"", "0", "false", "False"}

    # Optional client-side WebRTC VAD (sends only speech + explicit utterance boundaries).
    # Enabled with ARIA_CLIENT_VAD=1.
    vad = None
    vad_frame_ms = 20
    vad_frame_bytes = int(contract.sample_rate * 2 * vad_frame_ms / 1000)
    vad_silence_ms = int(os.environ.get("ARIA_CLIENT_VAD_SILENCE_MS", "300"))
    vad_aggr = int(os.environ.get("ARIA_CLIENT_VAD_AGGR", "2"))
    vad_pre_ms = int(os.environ.get("ARIA_CLIENT_VAD_PRE_MS", "200"))
    vad_start_frames = int(os.environ.get("ARIA_CLIENT_VAD_START_FRAMES", "3"))
    vad_min_rms = float(os.environ.get("ARIA_CLIENT_VAD_MIN_RMS", "0.0"))
    vad_snr_db = float(os.environ.get("ARIA_CLIENT_VAD_SNR_DB", "0.0"))
    vad_noise_alpha = float(os.environ.get("ARIA_CLIENT_VAD_NOISE_ALPHA", "0.05"))
    vad_pre_frames = max(0, int(vad_pre_ms / vad_frame_ms))
    vad_pending = bytearray()
    vad_preroll: collections.deque[bytes] = collections.deque(maxlen=vad_pre_frames)
    vad_in_utt = False
    vad_silence_acc = 0
    vad_speech_run = 0
    vad_send_buf = bytearray()
    vad_noise_rms = 0.0
    vad_snr_mult = 1.0
    if vad_snr_db > 0.0:
        vad_snr_mult = float(10.0 ** (vad_snr_db / 20.0))

    if client_vad_enabled:
        if webrtcvad is None:
            print(
                "ARIA: client VAD requested but webrtcvad is not installed; disabling ARIA_CLIENT_VAD",
                file=sys.stderr,
                flush=True,
            )
            client_vad_enabled = False
        else:
            try:
                vad = webrtcvad.Vad(int(vad_aggr))
            except Exception:
                vad = webrtcvad.Vad(2)

    # Gain strategy:
    # - If ARIA_CLIENT_GAIN is set, use it as a fixed multiplier.
    # - Otherwise use a conservative automatic gain control (AGC) to help very
    #   quiet mics trigger server VAD.
    fixed_gain_env = os.environ.get("ARIA_CLIENT_GAIN")
    fixed_gain: float | None = None
    if fixed_gain_env is not None and fixed_gain_env.strip() != "":
        try:
            fixed_gain = float(fixed_gain_env)
        except ValueError:
            fixed_gain = 1.0
        if fixed_gain <= 0:
            fixed_gain = 1.0

    # AGC can significantly raise the noise floor in quiet rooms, which can cause
    # client-side VAD to trigger continuously. Default AGC off when client VAD is on.
    default_agc = "0" if client_vad_enabled else "1"
    agc_enabled = os.environ.get("ARIA_CLIENT_AGC", default_agc).strip() not in {"0", "false", "False"}
    target_rms = float(os.environ.get("ARIA_CLIENT_TARGET_RMS", "0.05"))
    max_gain = float(os.environ.get("ARIA_CLIENT_MAX_GAIN", "50"))
    gain = 1.0
    sent_bytes = 0
    dropped_ws_chunks = 0
    last_stat = asyncio.get_running_loop().time()
    stat_pre_acc = 0.0
    stat_post_acc = 0.0
    stat_n = 0

    try:
        while True:
            if debug:
                now = asyncio.get_running_loop().time()
                if now - last_stat >= 1.0:
                    rms_pre = (stat_pre_acc / stat_n) ** 0.5 if stat_n else 0.0
                    rms_post = (stat_post_acc / stat_n) ** 0.5 if stat_n else 0.0
                    if client_vad_enabled:
                        pending_str = (
                            f"vad_in_utt={1 if vad_in_utt else 0} "
                            f"vad_pending={len(vad_pending)}B send_buf={len(vad_send_buf)}B "
                            f"silence_ms={vad_silence_acc} noise={vad_noise_rms:.4f}"
                        )
                    else:
                        pending_str = f"pending={len(out_bytes)}B"
                    print(
                        (
                            f"ARIA stats: sent={sent_bytes}B queued={queue.qsize()} {pending_str} "
                            f"rms_pre={rms_pre:.5f} rms_post={rms_post:.5f} "
                            f"gain={gain:.2f} dropped_ws_chunks={dropped_ws_chunks} "
                            f"enq={stats.enqueued_blocks} drop_q={stats.dropped_queue_blocks} "
                            f"client_vad={'on' if client_vad_enabled else 'off'} "
                            f"agc={'on' if agc_enabled and fixed_gain is None else 'off'} "
                            f"target={target_rms:g} max_gain={max_gain:g}"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    last_stat = now
                    stat_pre_acc = 0.0
                    stat_post_acc = 0.0
                    stat_n = 0

            if stop_event.is_set() and queue.empty():
                if not client_vad_enabled and len(out_bytes) < contract.bytes_per_chunk:
                    return

                if client_vad_enabled and len(vad_pending) < vad_frame_bytes and len(vad_send_buf) < contract.bytes_per_chunk:
                    if vad_in_utt:
                        # Flush any pending audio and close the utterance.
                        while len(vad_send_buf) >= contract.bytes_per_chunk:
                            chunk = bytes(vad_send_buf[: contract.bytes_per_chunk])
                            del vad_send_buf[: contract.bytes_per_chunk]
                            try:
                                await asyncio.wait_for(ws.send(chunk), timeout=0.05)
                                sent_bytes += len(chunk)
                            except asyncio.TimeoutError:
                                dropped_ws_chunks += 1

                        if vad_send_buf:
                            try:
                                await asyncio.wait_for(ws.send(bytes(vad_send_buf)), timeout=0.05)
                                sent_bytes += len(vad_send_buf)
                            except asyncio.TimeoutError:
                                dropped_ws_chunks += 1
                        vad_send_buf.clear()

                        try:
                            await asyncio.wait_for(ws.send(_send_msg({"type": "utterance_end"})), timeout=0.2)
                        except Exception:
                            pass

                    return
            try:
                block = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            # Resample float32 device audio -> 16kHz float32 (streaming).
            y = resampler.resample_chunk(block, last=False)
            y = np.asarray(y, dtype=np.float32)
            if y.size == 0:
                continue

            # Energy estimate pre-clip for AGC.
            block_rms_pre = float(np.sqrt(np.mean(np.square(y))))

            if fixed_gain is not None:
                gain = fixed_gain
            elif agc_enabled:
                # Smooth gain updates to avoid pumping.
                if block_rms_pre > 1e-6:
                    desired = target_rms / block_rms_pre
                    desired = float(np.clip(desired, 1.0, max_gain))
                    # Faster attack (increase gain) than release.
                    alpha = 0.25 if desired > gain else 0.08
                    gain = (1.0 - alpha) * gain + alpha * desired

            if gain != 1.0:
                y = y * gain

            if debug:
                stat_pre_acc += block_rms_pre * block_rms_pre
                stat_post_acc += float(np.mean(np.square(y)))
                stat_n += 1

            # Convert float32 [-1, 1] -> PCM16LE.
            y = np.clip(y, -1.0, 1.0)
            pcm16 = (y * 32767.0).astype(np.int16)
            pcm_bytes = pcm16.tobytes(order="C")

            if not client_vad_enabled:
                out_bytes.extend(pcm_bytes)

                # Ensure correct chunk boundaries after resampling: send fixed-size
                # frames (~40ms @ 16kHz by default).
                while len(out_bytes) >= contract.bytes_per_chunk:
                    chunk = bytes(out_bytes[: contract.bytes_per_chunk])
                    del out_bytes[: contract.bytes_per_chunk]
                    # Drop audio frames if websocket backpressure occurs (do not buffer unbounded).
                    try:
                        await asyncio.wait_for(ws.send(chunk), timeout=0.05)
                        sent_bytes += len(chunk)
                    except asyncio.TimeoutError:
                        dropped_ws_chunks += 1
                continue

            # Client-side VAD path: feed 20ms frames to webrtcvad and only send speech.
            vad_pending.extend(pcm_bytes)
            while len(vad_pending) >= vad_frame_bytes:
                frame = bytes(vad_pending[:vad_frame_bytes])
                del vad_pending[:vad_frame_bytes]

                try:
                    is_speech = bool(vad.is_speech(frame, contract.sample_rate)) if vad is not None else False
                except Exception:
                    is_speech = False

                # Compute frame RMS (0..1) and maintain an adaptive noise floor estimate.
                samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                frame_rms = float(np.sqrt(np.mean(samples * samples)) / 32768.0) if samples.size else 0.0
                if not vad_in_utt:
                    a = float(np.clip(vad_noise_alpha, 0.0, 1.0))
                    if vad_noise_rms <= 0.0:
                        vad_noise_rms = frame_rms
                    else:
                        vad_noise_rms = (1.0 - a) * vad_noise_rms + a * frame_rms

                energy_ok = True

                # Optional RMS floor to avoid false positives from very low-level noise.
                if vad_min_rms > 0.0:
                    if frame_rms < vad_min_rms:
                        energy_ok = False

                # Optional SNR gating relative to noise floor.
                if energy_ok and vad_snr_db > 0.0:
                    denom = vad_noise_rms if vad_noise_rms > 1e-6 else 1e-6
                    if (frame_rms / denom) < vad_snr_mult:
                        energy_ok = False

                is_speech_eff = bool(is_speech) and energy_ok

                if not vad_in_utt:
                    vad_preroll.append(frame)
                    if is_speech_eff:
                        vad_speech_run += 1
                    else:
                        vad_speech_run = 0

                    # Require N consecutive speech frames before declaring start.
                    if vad_speech_run < max(1, vad_start_frames):
                        continue

                    vad_in_utt = True
                    vad_speech_run = 0
                    vad_silence_acc = 0
                    try:
                        await asyncio.wait_for(ws.send(_send_msg({"type": "utterance_start"})), timeout=0.2)
                    except Exception:
                        pass
                    for fr in list(vad_preroll):
                        vad_send_buf.extend(fr)
                    vad_preroll.clear()
                else:
                    vad_send_buf.extend(frame)
                    if is_speech_eff:
                        vad_silence_acc = 0
                    else:
                        vad_silence_acc += vad_frame_ms

                while len(vad_send_buf) >= contract.bytes_per_chunk:
                    chunk = bytes(vad_send_buf[: contract.bytes_per_chunk])
                    del vad_send_buf[: contract.bytes_per_chunk]
                    try:
                        await asyncio.wait_for(ws.send(chunk), timeout=0.05)
                        sent_bytes += len(chunk)
                    except asyncio.TimeoutError:
                        dropped_ws_chunks += 1

                if vad_in_utt and vad_silence_acc >= vad_silence_ms:
                    if vad_send_buf:
                        try:
                            await asyncio.wait_for(ws.send(bytes(vad_send_buf)), timeout=0.05)
                            sent_bytes += len(vad_send_buf)
                        except asyncio.TimeoutError:
                            dropped_ws_chunks += 1
                    vad_send_buf.clear()
                    try:
                        await asyncio.wait_for(ws.send(_send_msg({"type": "utterance_end"})), timeout=0.2)
                    except Exception:
                        pass
                    vad_in_utt = False
                    vad_silence_acc = 0
    except asyncio.CancelledError:
        raise
    except Exception:
        # Connection likely closed; request shutdown.
        stop_event.set()
        return


async def run_client(url: str, contract: AriaAudioContract, *, device: str | None) -> int:
    ws_url = _normalize_ws_url(url)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=50)
    stats = ClientStats()
    dropped = 0

    def request_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:
            # Some environments don't support signal handlers.
            pass

    def _enqueue_block(block: np.ndarray) -> None:
        nonlocal dropped
        if stop_event.is_set():
            return
        try:
            queue.put_nowait(block)
            stats.enqueued_blocks += 1
        except asyncio.QueueFull:
            dropped += 1
            stats.dropped_queue_blocks += 1

    def audio_callback(indata: np.ndarray, frames: int, _time: Any, status: sd.CallbackFlags) -> None:
        if stop_event.is_set():
            return
        if status:
            # Over/under-runs happen; we prefer dropping over blocking.
            pass

        # Ensure mono float32 (sounddevice provides float32 when configured).
        data = indata
        if data.ndim == 2:
            data = data[:, 0]
        block = np.asarray(data, dtype=np.float32).copy()
        loop.call_soon_threadsafe(_enqueue_block, block)

    # Query the default input device's native sample rate (do NOT force 16k at capture).
    selected_device = _select_input_device(device)
    if selected_device is None:
        try:
            selected_device = sd.default.device[0]  # (input, output)
        except Exception:
            selected_device = None

    device_info = sd.query_devices(selected_device, "input")
    in_sample_rate = int(round(float(device_info["default_samplerate"])))

    print(f"ARIA Capturing from: {device_info.get('name', 'unknown')} @ {in_sample_rate} Hz", flush=True)

    # Capture in float32 at the device rate; we will resample to 16kHz for transport.
    in_blocksize = max(1, int(round(in_sample_rate * contract.chunk_ms / 1000)))

    client_vad_requested = _truthy_env("ARIA_CLIENT_VAD", "0")
    client_vad_enabled = client_vad_requested and webrtcvad is not None
    if client_vad_requested and not client_vad_enabled:
        print(
            "ARIA: ARIA_CLIENT_VAD=1 but webrtcvad is not installed in this environment; running without client VAD",
            file=sys.stderr,
            flush=True,
        )

    start_msg = {
        "type": "start",
        "sample_rate": contract.sample_rate,
        "channels": contract.channels,
        "format": contract.fmt,
        "chunk_ms": contract.chunk_ms,
        "client_vad": bool(client_vad_enabled),
    }

    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps(start_msg))

        recv_task = asyncio.create_task(_recv_loop(ws, stop_event))
        send_task = asyncio.create_task(
            _send_loop(
                ws,
                queue,
                stop_event,
                in_sample_rate=in_sample_rate,
                contract=contract,
                stats=stats,
                client_vad_enabled=bool(client_vad_enabled),
            )
        )

        # Start microphone capture.
        stream = sd.InputStream(
            samplerate=in_sample_rate,
            channels=contract.channels,
            dtype="float32",
            blocksize=in_blocksize,
            device=selected_device,
            callback=audio_callback,
        )

        try:
            with stream:
                while not stop_event.is_set():
                    await asyncio.sleep(0.1)
        finally:
            stop_event.set()
            try:
                await ws.send(json.dumps({"type": "stop"}))
            except Exception:
                pass

            send_task.cancel()
            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await send_task
            with contextlib.suppress(asyncio.CancelledError):
                await recv_task

    if dropped:
        print(f"Dropped {dropped} audio chunks due to backpressure", file=sys.stderr)
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mic_stream.py", description="ARIA macOS mic streamer")
    p.add_argument("url", help="Server URL, e.g. ws://host:8000/ws/asr")
    p.add_argument("--chunk-ms", type=int, default=40, help="Chunk size in ms (20-60 recommended)")
    p.add_argument(
        "--device",
        default=None,
        help="Optional input device (PortAudio index or name substring). Default: system input.",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    contract = AriaAudioContract(chunk_ms=args.chunk_ms)
    try:
        return asyncio.run(run_client(args.url, contract, device=args.device))
    except KeyboardInterrupt:
        return 0
    except asyncio.CancelledError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
