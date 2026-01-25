import argparse
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf


def record_wav(out_path: Path, seconds: float = 5.0, samplerate: int = 48000) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording {seconds:.1f}s to {out_path} ...")
    sd.default.samplerate = samplerate
    sd.default.channels = 1

    audio = sd.rec(int(seconds * samplerate), dtype="float32")
    time.sleep(seconds)
    sd.wait()

    sf.write(str(out_path), audio, samplerate)
    print("Saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record speaker samples (client-side)")
    parser.add_argument("name", nargs="?", default="speaker", help="Speaker name")
    parser.add_argument("--count", type=int, default=3, help="Number of samples")
    parser.add_argument("--seconds", type=float, default=5.0, help="Seconds per sample")
    parser.add_argument("--samplerate", type=int, default=48000, help="Sample rate")
    parser.add_argument("--out", default="samples", help="Output base directory")
    args = parser.parse_args()

    base = Path(args.out) / args.name
    for i in range(args.count):
        input(f"\nPress ENTER to record sample {i+1}/{args.count}...")
        record_wav(base / f"{args.name}_{i+1}.wav", seconds=args.seconds, samplerate=args.samplerate)

    print("\nDone. You should now have samples in:", base)
