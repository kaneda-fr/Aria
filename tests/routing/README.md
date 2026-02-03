# Intent Routing Benchmark

This directory contains tools for testing embedding-based intent routing (chat vs control classification).

## Quick Start (ONNX INT8 - Recommended)

### 1. Install Dependencies

```bash
pip install sentence-transformers scikit-learn
pip install optimum==1.19.2 onnx onnxruntime  # For ONNX quantization
```

### 2. Generate Test Dataset

```bash
# Generate 1000 examples with edge cases
python3 tests/routing/generate_dataset.py \
    --count 1000 \
    --output data/routing/synthetic_with_edges.jsonl \
    --edge-case-rate 0.4 \
    --typo-rate 0.2
```

### 3. Convert Model to ONNX INT8

```bash
# One-time conversion for 4-7x speedup
python3 tests/routing/convert_to_onnx.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output models/all-MiniLM-L6-v2-onnx-int8
```

### 4. Run Benchmark

```bash
# Compare PyTorch vs ONNX INT8
python3 tests/routing/benchmark_onnx.py \
    --pytorch-model sentence-transformers/all-MiniLM-L6-v2 \
    --onnx-model models/all-MiniLM-L6-v2-onnx-int8 \
    --dataset data/routing/synthetic_with_edges.jsonl \
    --output results/onnx_comparison.csv
```

## Results Summary

| Model | Accuracy | P50 Latency | Speedup |
|-------|----------|-------------|---------|
| PyTorch L6-v2 | 94.5% | 7.1ms | 1x |
| **ONNX INT8 L6-v2** | **98.5%** | **1.6ms** | **4.4x** ✅ |

## Dataset Format

Each line is a JSON object with:
```json
{
  "text": "allume la lumière du salon",
  "label": "control",  // or "chat"
  "lang": "fr"  // or "en"
}
```

## Benchmark Output

The benchmark will output:
1. **Console table** - Comparison of models with metrics
2. **CSV file** - Detailed results for plotting

### Metrics Explained

- **Precision**: Of all "control" predictions, what % were correct?
- **Recall**: Of all actual "control" examples, what % did we catch?
- **F1 Score**: Harmonic mean of precision and recall (overall accuracy)
- **FP Rate**: False positive rate (chat → control mistakes)
- **FN Rate**: False negative rate (control → chat mistakes)
- **Latency P50/P95**: Median and 95th percentile latency in milliseconds

## Example Output

```
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

## Files

- `generate_dataset.py` - Synthetic dataset generator with edge cases
- `convert_to_onnx.py` - ONNX INT8 quantization ✅ **Use this**
- `benchmark_onnx.py` - PyTorch vs ONNX comparison ✅ **Recommended**
- `benchmark_v2.py` - PyTorch-only benchmark
- `benchmark_lang_aware.py` - Language-aware routing test
- `analyze_errors.py` - Error analysis tool
- `test_edge_cases.py` - Edge case validator
- `quantize_manual.py` - Alternative ONNX conversion (if optimum fails)

## Production Deployment

After benchmarking, deploy with ONNX INT8 model:

```bash
# Set environment variables
export ARIA_EMBEDDING_ROUTER_ENABLED=1
export ARIA_EMBEDDING_ROUTER_MODEL=models/all-MiniLM-L6-v2-onnx-int8
export ARIA_EMBEDDING_ROUTER_THRESHOLD=0.5
export HF_HOME=/path/to/project/.cache/huggingface
export TRANSFORMERS_CACHE=/path/to/project/.cache/huggingface/transformers

# Start Aria
python -m app.main
```

**Expected Performance:**
- 98.5% accuracy on realistic edge cases
- 1.6ms P50 latency (6x faster than 10ms target)
- 100% recall (catches ALL control commands)
- 3% false positive rate

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for detailed analysis.
