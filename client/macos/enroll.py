import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference import EncoderClassifier


def load_mono_16k(wav_path: Path) -> torch.Tensor:
    print(f"Loading audio: {wav_path}")
    wav, sr = torchaudio.load(str(wav_path))
    print(f"  sample rate: {sr}, channels: {wav.shape[0]}, samples: {wav.shape[1]}")
    wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != 16000:
        print("  resampling to 16kHz")
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=0)


def enroll_from_wavs(name: str, wav_files: list[Path], out_dir: Path, device: str = "cpu") -> None:
    print(f"Starting enrollment for '{name}'")
    print("Loading speaker encoder model...")
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    print("Model loaded.")

    if not wav_files:
        raise SystemExit(f"No wav files found for '{name}'")
    print(f"Found {len(wav_files)} wav files.")

    embs = []
    for idx, p in enumerate(wav_files, start=1):
        print(f"[{idx}/{len(wav_files)}] Processing {p.name}")
        wav = load_mono_16k(p).to(device)
        with torch.no_grad():
            emb = clf.encode_batch(wav)
        emb = emb.squeeze().detach().cpu()
        emb = l2_normalize(emb)
        embs.append(emb)
        print(f"  embedding shape: {tuple(emb.shape)}")

    print("Averaging embeddings...")
    profile = torch.stack(embs, dim=0).mean(dim=0)
    profile = l2_normalize(profile).numpy().astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps({"name": name, "embedding": profile.tolist()}), encoding="utf-8")
    print(f"Saved profile: {out_path}")
    print(f"Enrollment complete for '{name}'.")


def enroll(name: str, samples_dir: Path, out_dir: Path, device: str = "cpu") -> None:
    print(f"Scanning samples in: {samples_dir}")
    wav_files = sorted(samples_dir.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"No wav files found in {samples_dir}")
    enroll_from_wavs(name=name, wav_files=wav_files, out_dir=out_dir, device=device)


def enroll_all(samples_root: Path, out_dir: Path, device: str = "cpu") -> None:
    speaker_dirs = sorted([p for p in samples_root.iterdir() if p.is_dir()])
    if not speaker_dirs:
        raise SystemExit(f"No speaker folders found in {samples_root}")
    for speaker_dir in speaker_dirs:
        name = speaker_dir.name
        enroll(name=name, samples_dir=speaker_dir, out_dir=out_dir, device=device)


def enroll_auto(samples_root: Path, out_dir: Path, device: str = "cpu") -> None:
    speaker_dirs = sorted([p for p in samples_root.iterdir() if p.is_dir()])
    if speaker_dirs:
        enroll_all(samples_root=samples_root, out_dir=out_dir, device=device)
        return

    wav_files = sorted(samples_root.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"No wav files found in {samples_root}")

    groups: dict[str, list[Path]] = {}
    for wav in wav_files:
        stem = wav.stem
        if "_" in stem:
            name = stem.split("_", 1)[0]
        else:
            name = "speaker"
        groups.setdefault(name, []).append(wav)

    if len(groups) == 1 and "speaker" in groups:
        print("Auto-detected samples without prefixes; enrolling as 'speaker'.")
    else:
        print(f"Auto-detected speakers: {', '.join(sorted(groups.keys()))}")

    for name, files in sorted(groups.items()):
        enroll_from_wavs(name=name, wav_files=files, out_dir=out_dir, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll speaker profiles (client-side)")
    parser.add_argument("name", nargs="?", help="Speaker name (enroll single user)")
    parser.add_argument("--all", action="store_true", help="Enroll all speakers under samples/")
    parser.add_argument("--auto", action="store_true", help="Auto-detect speakers and enroll")
    parser.add_argument("--samples", default="samples", help="Samples root directory")
    parser.add_argument("--out", default="profiles", help="Output profiles directory")
    parser.add_argument("--device", default="cpu", help="Device to run model on")
    args = parser.parse_args()

    samples_root = Path(args.samples)
    out_dir = Path(args.out)

    if args.all:
        enroll_all(samples_root=samples_root, out_dir=out_dir, device=args.device)
    elif args.auto or args.name is None:
        enroll_auto(samples_root=samples_root, out_dir=out_dir, device=args.device)
    else:
        name = args.name
        enroll(
            name=name,
            samples_dir=samples_root / name,
            out_dir=out_dir,
            device=args.device,
        )
