from pathlib import Path
import soundfile as sf
import onnx_asr

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = str((ROOT / "models/parakeet-tdt-0.6b-v3-onnx").resolve())
WAV_PATH = (ROOT / "test.wav").resolve()

print("Repo root:", ROOT)
print("WAV:", WAV_PATH)

audio, sr = sf.read(str(WAV_PATH), dtype="float32")
if audio.ndim > 1:
    audio = audio[:, 0]
if sr != 16000:
    print(f"Warning: expected 16kHz wav, got {sr} Hz; onnx-asr will handle input via file path.")

model = onnx_asr.load_model(
    "nemo-parakeet-tdt-0.6b-v3",
    MODEL_DIR,
    providers=["CPUExecutionProvider"],
)

text = model.recognize(str(WAV_PATH))
print("\nTRANSCRIPT:\n", text)
