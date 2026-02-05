# Running Trilingual Benchmark on Server

## Server: 192.168.100.100 (Xeon E3)

### Quick Start

#### 1. Sync code to server
```bash
# On your local machine
./sync.sh
```

#### 2. SSH to server
```bash
ssh user@192.168.100.100
cd /path/to/Aria
```

#### 3. Install dependencies (one-time)
```bash
pip install sentence-transformers scikit-learn
pip install optimum==1.19.2 onnx onnxruntime
```

#### 4. Run benchmark
```bash
./tests/routing/run_benchmark_server.sh
```

This will:
- Generate 1000 trilingual examples (FR + EN + DE)
- Convert model to ONNX INT8 (if not already done)
- Benchmark PyTorch vs ONNX INT8
- Save results to `results/trilingual_onnx_comparison.csv`

### Expected Output

```
==================================================
Aria Trilingual Intent Routing Benchmark
French + English + German
==================================================

Step 1: Generate trilingual test dataset
  Size: 1000 examples (FR + EN + DE)
  ...
✓ Generated 1000 examples
  - Control: 500
  - Chat: 500
  - French: ~333
  - English: ~333
  - German: ~334

Step 2: Convert model to ONNX INT8
  ...
✓ Conversion complete!
  Model saved to: models/all-MiniLM-L6-v2-onnx-int8
  Model size: 22.0 MB

Step 3: Run benchmark (PyTorch vs ONNX INT8)
  ...

PyTorch vs ONNX-INT8 Comparison
============================================================================
Model                          Type        Acc   Prec  Recall  F1    FP    FN    P50    P95
----------------------------------------------------------------------------
all-MiniLM-L6-v2              pytorch    0.945  0.941  0.950  0.945  6%    5%    7.1ms  7.8ms
all-MiniLM-L6-v2-onnx-int8    onnx-int8  0.985  0.971  1.000  0.985  3%    0%    1.6ms  2.3ms
============================================================================

⚡ ONNX INT8 Speedup: 4.4x faster
   Accuracy: +4.0% better
```

### Manual Step-by-Step (Alternative)

If you want to run each step manually:

```bash
# Set cache directories
export HF_HOME="$(pwd)/.cache/huggingface"
export TRANSFORMERS_CACHE="$(pwd)/.cache/huggingface/transformers"

# 1. Generate dataset
python3 tests/routing/generate_dataset.py \
    --count 1000 \
    --output data/routing/synthetic_trilingual_edges.jsonl \
    --edge-case-rate 0.4 \
    --typo-rate 0.2

# 2. Convert to ONNX INT8
python3 tests/routing/convert_to_onnx.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output models/all-MiniLM-L6-v2-onnx-int8

# 3. Run benchmark
python3 tests/routing/benchmark_onnx.py \
    --pytorch-model sentence-transformers/all-MiniLM-L6-v2 \
    --onnx-model models/all-MiniLM-L6-v2-onnx-int8 \
    --dataset data/routing/synthetic_trilingual_edges.jsonl \
    --output results/trilingual_onnx_comparison.csv
```

### Viewing Results

```bash
# View CSV results
cat results/trilingual_onnx_comparison.csv

# View German examples
cat data/routing/synthetic_trilingual_edges.jsonl | grep '"lang": "de"' | head -10

# Detailed analysis by language and intent type
python3 tests/routing/analyze_by_language.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --dataset data/routing/synthetic_trilingual_edges.jsonl \
    --show-errors
```

This will show:
- Overall performance metrics
- Per-language breakdown (FR, EN, DE)
- Per-intent breakdown (control, chat)
- Detailed language × intent matrix
- Sample misclassifications per language

See [LANGUAGE_ANALYSIS_EXAMPLE.md](LANGUAGE_ANALYSIS_EXAMPLE.md) for expected output and interpretation guide.

### Production Deployment

After successful benchmark, deploy with:

```bash
# Add to your startup script or .bashrc
export ARIA_EMBEDDING_ROUTER_ENABLED=1
export ARIA_EMBEDDING_ROUTER_MODEL="$(pwd)/models/all-MiniLM-L6-v2-onnx-int8"
export ARIA_EMBEDDING_ROUTER_THRESHOLD=0.5
export HF_HOME="$(pwd)/.cache/huggingface"
export TRANSFORMERS_CACHE="$(pwd)/.cache/huggingface/transformers"

# Start Aria
python -m app.main
```

### Troubleshooting

#### Permission denied on cache folder
- The script automatically sets `HF_HOME` to a local directory
- If issues persist, check permissions: `chmod -R 755 .cache/`

#### Out of memory
- Close other applications
- Reduce dataset size: `--count 500` instead of 1000

#### Model download fails
- Check internet connection
- Pre-download on local machine and sync with `./sync.sh`

#### ONNX conversion fails
- Try alternative: `python3 tests/routing/quantize_manual.py`
- Or use PyTorch-only benchmark: `python3 tests/routing/benchmark_v2.py`
