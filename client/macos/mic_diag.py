from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd


def main() -> int:
    p = argparse.ArgumentParser(description="ARIA mic capture diagnostic")
    p.add_argument("--seconds", type=float, default=3.0, help="Record duration")
    args = p.parse_args()

    default_in = None
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None

    info = sd.query_devices(default_in, "input")
    sr = int(round(float(info["default_samplerate"])))
    print(f"Device: {info.get('name', 'unknown')}")
    print(f"Native sample rate: {sr} Hz")

    blocksize = max(1, int(sr * 0.04))
    chunks: list[np.ndarray] = []

    def cb(indata: np.ndarray, _frames: int, _time: sd.CallbackFlags, status: sd.CallbackFlags) -> None:
        if status:
            pass
        x = indata
        if x.ndim == 2:
            x = x[:, 0]
        chunks.append(np.asarray(x, dtype=np.float32).copy())

    print(f"Recording {args.seconds:.1f}s... speak now")
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=blocksize, callback=cb):
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            time.sleep(0.05)

    x = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
    if x.size == 0:
        print("Captured 0 samples (no audio received)")
        return 2

    rms = float(np.sqrt(np.mean(np.square(x))))
    peak = float(np.max(np.abs(x)))
    print(f"Captured samples: {x.size}")
    print(f"RMS:  {rms:.6f}")
    print(f"Peak: {peak:.6f}")

    if peak < 0.01:
        print("Result: looks like near-silence (wrong device/muted/permission issue)")
    else:
        print("Result: capture looks OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
