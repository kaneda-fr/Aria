from pathlib import Path
import onnxruntime as ort

MODEL_DIR = Path("models/parakeet-tdt-0.6b-v3-onnx").resolve()

enc = MODEL_DIR / "encoder-model.onnx"
dec = MODEL_DIR / "decoder_joint-model.onnx"

print("Using model dir:", MODEL_DIR)
print("Available providers:", ort.get_available_providers())

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.inter_op_num_threads = 1

# Force CPU only to avoid CoreML provider path
providers = ["CPUExecutionProvider"]

print("Loading encoder:", enc)
enc_sess = ort.InferenceSession(str(enc), sess_options=so, providers=providers)
print("Encoder providers:", enc_sess.get_providers())

print("Loading decoder:", dec)
dec_sess = ort.InferenceSession(str(dec), sess_options=so, providers=providers)
print("Decoder providers:", dec_sess.get_providers())

print("✅ ORT CPU-only sessions created successfully.")
