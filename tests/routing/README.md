# Intent Routing Benchmark

This directory contains tools for testing embedding-based intent routing (chat vs control classification).

## Quick Start

### 1. Install Dependencies

```bash
pip install sentence-transformers onnxruntime scikit-learn
```

### 2. Generate Test Dataset

```bash
# Generate 200 examples for quick testing
python3 tests/routing/generate_dataset.py --count 200 --output data/routing/test.jsonl

# Generate full 1000 example dataset
python3 tests/routing/generate_dataset.py --count 1000 --output data/routing/synthetic_examples.jsonl
```

### 3. Run Benchmark

```bash
# Test single fast model
python3 tests/routing/benchmark.py \
    --models all-MiniLM-L6-v2 \
    --dataset data/routing/test.jsonl \
    --output results/quick_test.csv

# Compare multiple models
python3 tests/routing/benchmark.py \
    --models all-MiniLM-L6-v2,all-MiniLM-L12-v2,paraphrase-multilingual-MiniLM-L12-v2 \
    --dataset data/routing/synthetic_examples.jsonl \
    --output results/baseline_benchmark.csv
```

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
Model Comparison - Embedding-Based Intent Routing
================================================================================
Model                               Thresh   Prec  Recall     F1  FP Rate  FN Rate  P50 (ms)  P95 (ms)
--------------------------------------------------------------------------------
all-MiniLM-L6-v2                      0.60   0.91    0.87   0.89     0.04     0.08       4.2       8.1
all-MiniLM-L12-v2                     0.55   0.93    0.89   0.91     0.03     0.07      12.5      18.3
paraphrase-multilingual-MiniLM...    0.50   0.95    0.92   0.93     0.02     0.05      18.7      24.9
================================================================================
```

## Files

- `generate_dataset.py` - Synthetic dataset generator
- `benchmark.py` - Model evaluation harness
- `README.md` - This file

## Next Steps

After benchmarking, you can:
1. Choose the best model based on your latency vs accuracy tradeoff
2. Integrate the router into `app/main.py` (see implementation plan)
3. Fine-tune the model with real production data
4. Deploy with environment variables:
   ```bash
   export ARIA_EMBEDDING_ROUTER_ENABLED=1
   export ARIA_EMBEDDING_ROUTER_MODEL=all-MiniLM-L6-v2
   export ARIA_EMBEDDING_ROUTER_THRESHOLD=0.60
   ```
