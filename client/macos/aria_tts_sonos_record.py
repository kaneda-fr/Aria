import argparse
import http.server
import os
import socket
import threading
import time
from pathlib import Path
from functools import partial

import sounddevice as sd
import soundfile as sf

try:
    import soco
except Exception as exc:  # pragma: no cover
    soco = None
    _soco_err = exc
else:
    _soco_err = None

import subprocess


def _parse_voices(raw: str) -> dict[str, Path]:
    voices: dict[str, Path] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
            name = name.strip()
            path = path.strip()
        else:
            path = item
            name = Path(path).stem
        if not name or not path:
            continue
        voices[name] = Path(path)
    return voices


def _local_ip_for(dest_ip: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((dest_ip, 80))
        return sock.getsockname()[0]
    finally:
        sock.close()


def _run_piper(piper_bin: str, model_path: Path, text: str, out_wav: Path) -> None:
    """
    Generate TTS WAV using the piper-tts Python module.
    Assumes model_path is the .onnx file; looks for .json config in the same directory.
    """
    import piper
    model_path = Path(model_path)
    if model_path.suffix != ".onnx":
        raise ValueError(f"Model path must be a .onnx file: {model_path}")
    # Prefer .onnx.json, fallback to .json
    config_path_onnx_json = model_path.with_suffix(".onnx.json")
    config_path_json = model_path.with_suffix(".json")
    if config_path_onnx_json.exists():
        config_path = config_path_onnx_json
    elif config_path_json.exists():
        config_path = config_path_json
    else:
        raise FileNotFoundError(f"Config file not found: {config_path_onnx_json} or {config_path_json}")
    import wave
    tts = piper.PiperVoice.load(str(model_path), str(config_path))
    with wave.open(str(out_wav), "wb") as wf:
        tts.synthesize_wav(text, wf)


def _normalize_wav(ffmpeg_bin: str, in_wav: Path, out_wav: Path) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(in_wav),
        "-ac",
        "2",
        "-ar",
        "44100",
        "-sample_fmt",
        "s16",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg normalization failed for {in_wav}")


def _wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def _record_to_file(path: Path, seconds: float, samplerate: int = 48000) -> None:
    sd.default.samplerate = samplerate
    sd.default.channels = 1
    audio = sd.rec(int(seconds * samplerate), dtype="float32")
    sd.wait()
    sf.write(str(path), audio, samplerate)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate TTS WAVs, play on Sonos, and record mic on macOS.")
    parser.add_argument("--voices", required=True, help="Comma list: name=/path/model.onnx or /path/model.onnx")
    parser.add_argument("--text", default="Bonjour, je m'appelle ARIA.", help="Text to speak for all voices (overridden by --texts)")
    parser.add_argument("--texts", default=None, help="Comma list: name=text. E.g. en=Hello!,fr=Bonjour!")
    parser.add_argument("--out", default="samples", help="Output base directory (recordings saved here)")
    parser.add_argument("--sonos-ip", required=True, help="Sonos speaker IP")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="Path to ffmpeg binary")
    parser.add_argument("--normalize", action="store_true", help="Normalize WAV for Sonos (ffmpeg)")
    parser.add_argument("--http-port", type=int, default=18080, help="Port for local HTTP server")
    parser.add_argument("--tail-sec", type=float, default=0.4, help="Extra recording tail seconds")
    args = parser.parse_args()

    # Parse per-voice texts if provided (semicolon-separated to allow commas in text)
    texts = {}
    if args.texts:
        for item in args.texts.split(";"):
            if not item.strip():
                continue
            if "=" in item:
                name, txt = item.split("=", 1)
                texts[name.strip()] = txt.strip()

    if soco is None:
        raise SystemExit(f"Missing SoCo dependency: {_soco_err}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    voices = _parse_voices(args.voices)
    if not voices:
        raise SystemExit("No voices parsed. Use --voices name=/path/model.onnx,...")

    local_ip = _local_ip_for(args.sonos_ip)
    base_url = f"http://{local_ip}:{args.http_port}"

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(out_dir))
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", args.http_port), handler)

    def _serve() -> None:
        httpd.serve_forever()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    speaker = soco.SoCo(args.sonos_ip)

    for name, model_path in voices.items():
        if not model_path.exists():
            raise SystemExit(f"Model not found: {model_path}")

        # Use per-voice text if provided, else fallback to --text
        text = texts.get(name, args.text)

        raw_path = out_dir / f"aria_{name}_tts_raw.wav"
        out_path = out_dir / f"aria_{name}_tts.wav"
        rec_path = out_dir / f"aria_{name}_recorded.wav"

        print(f"Generating TTS for voice '{name}' -> {out_path}")
        print(f"  Text: {text}")
        _run_piper(None, model_path, text, raw_path)
        if args.normalize:
            _normalize_wav(args.ffmpeg_bin, raw_path, out_path)
        else:
            raw_path.replace(out_path)

        url = f"{base_url}/{out_path.name}"
        dur = _wav_duration(out_path)
        total = dur + args.tail_sec

        print(f"Playing on Sonos: {url} ({dur:.2f}s)")
        speaker.play_uri(url)
        time.sleep(0.1)

        print(f"Recording mic to: {rec_path} ({total:.2f}s)")
        _record_to_file(rec_path, total)

        time.sleep(0.2)

    httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
