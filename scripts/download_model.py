from huggingface_hub import snapshot_download

DEST = "models/parakeet-tdt-0.6b-v3-onnx"
repo_id = "istupakov/parakeet-tdt-0.6b-v3-onnx"

path = snapshot_download(
    repo_id=repo_id,
    local_dir=DEST,
    local_dir_use_symlinks=False,
)
print("Downloaded snapshot to:", path)
print("Model directory:", DEST)
