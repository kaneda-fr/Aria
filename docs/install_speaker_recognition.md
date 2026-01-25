# Speaker Recognition Install (Server)

## Requirements
- Python venv activated (e.g., `.venv`)
- Internet access to download PyTorch wheels and the SpeechBrain model

## Install (recommended)
Run:
- `python -m pip install -e .`

This installs:
- `speechbrain`
- `torch` + `torchaudio`

## CPU-only install (recommended)
If you want CPU-only wheels (avoid NVIDIA/CUDA downloads), run:
- `python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`

Optional (only if `ARIA_TTS_LANG_DETECT=langid`):
- `python -m pip install langid`

## After install
Restart the server so the speaker recognizer loads the model and profiles.
