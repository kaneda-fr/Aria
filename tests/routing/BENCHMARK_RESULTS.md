# Intent Routing Benchmark Results

This document describes the embedding-based intent routing test system and benchmark results.

## Overview

Testing framework to evaluate embedding models for binary classification (chat vs control) with focus on:
- High reliability for control detection
- Minimum latency on Xeon E3 CPU
- Support for French and English

## Quick Start

### 1. Install Dependencies

```bash
pip install sentence-transformers onnxruntime scikit-learn
```

### 2. Generate Test Dataset

```bash
# Generate 1000 examples with 40% edge cases
python3 tests/routing/generate_dataset.py \
    --count 1000 \
    --output data/routing/synthetic_with_edges.jsonl \
    --edge-case-rate 0.4 \
    --typo-rate 0.2
```

### 3. Run Benchmark

```bash
# Test with trained classifiers (recommended)
python3 tests/routing/benchmark_v2.py \
    --models all-MiniLM-L6-v2,all-MiniLM-L12-v2 \
    --dataset data/routing/synthetic_with_edges.jsonl \
    --output results/benchmark.csv
```

## Test Approaches

### Version 1: Reference Embeddings (benchmark.py)
- Simple cosine similarity against reference embeddings
- Fast but inaccurate (56% precision, 68% FP rate)
- ❌ Not recommended for production

### Version 2: Trained Logistic Regression (benchmark_v2.py)
- Trains sklearn LogisticRegression on embeddings
- 80/20 train/test split
- ✅ **Recommended approach**

### Language-Aware Routing (benchmark_lang_aware.py)
- Separate classifiers per language (French/English)
- Tests if language-specific routing improves accuracy
- Results: No significant improvement over multilingual

## Benchmark Results

### Dataset Characteristics

**Synthetic Dataset (1000 examples):**
- 50% control commands, 50% chat queries
- 50% French, 50% English
- 40% edge cases (implicit commands, status queries, minimal commands)
- 20% with typos

**Edge Case Types:**

Control edge cases:
- Implicit: `"il fait noir"` (it's dark)
- Minimal: `"lights"` (single word)
- Contextual: `"peux-tu m'aider avec les lumières"` (can you help with lights)

Chat edge cases:
- Status queries: `"are the lights on?"`
- Info queries: `"what lights do I have?"`

### Performance Results

#### Trained Classifier (benchmark_v2.py)

Test set: 200 examples (20% hold-out)

| Model | Accuracy | Precision | Recall | F1 | FP Rate | FN Rate | P50 (ms) | P95 (ms) |
|-------|----------|-----------|--------|----|---------|---------|---------:|--------:|
| **all-MiniLM-L6-v2** | 94% | 93% | 96% | 0.95 | 7% | 4% | 6.6 | 7.5 |
| **all-MiniLM-L12-v2** | 97% | 97% | 97% | 0.97 | 3% | 3% | 12.0 | 14.4 |

**Metrics Explained:**
- **Precision**: Of all "control" predictions, what % were correct?
- **Recall**: Of all actual "control" examples, what % were caught?
- **FP Rate**: False positive rate (chat → control mistakes) ⚠️ CRITICAL
- **FN Rate**: False negative rate (control → chat mistakes)
- **Latency**: Includes embedding generation (bottleneck)

#### Language-Aware vs Multilingual

| Model | Approach | F1 | FP Rate | Latency P50 |
|-------|----------|----|---------|-----------:|
| L6-v2 | Multilingual | **0.975** ✓ | 3.0% | 6.6ms |
| L6-v2 | Language-aware | 0.965 | 3.0% | 6.7ms |
| L12-v2 | Multilingual | **0.975** ✓ | 4.0% | 12.0ms |
| L12-v2 | Language-aware | 0.970 | **3.0%** ✓ | 12.0ms |

**Conclusion:** Multilingual approach performs better overall. Language-aware routing provides no significant benefit because:
- Sentence-transformers already learn language-invariant representations
- Chat vs control patterns are similar across languages
- Splitting training data reduces samples per classifier

## Production Recommendation

### Use: all-MiniLM-L12-v2 (Multilingual)

```bash
export ARIA_EMBEDDING_ROUTER_ENABLED=1
export ARIA_EMBEDDING_ROUTER_MODEL=all-MiniLM-L12-v2
export ARIA_EMBEDDING_ROUTER_THRESHOLD=0.5
```

**Rationale:**
- ✅ 97% accuracy on realistic edge cases
- ✅ 3% false positive rate (acceptable with monitoring)
- ✅ 12ms P50, 14ms P95 latency (meets <20ms target)
- ✅ Handles French + English together
- ✅ Simpler than language-aware (one classifier)

### Conservative Deployment

For first week, use higher threshold to reduce false positives:

```bash
ARIA_EMBEDDING_ROUTER_THRESHOLD=0.6  # More conservative
```

This will route fewer ambiguous queries to control, reducing risk of accidental device activation.

## Expected Production Performance

**Scenario:** 1000 user queries per day

- **970 correct routings** ✓
- **~30 errors total:**
  - 15 chat → control (false device activations) ⚠️
  - 15 control → chat (missed commands - user can repeat)

**Mitigation:**
- Conservative threshold (0.6) reduces FP to ~1%
- Log all routing decisions for review
- Collect misclassifications for monthly retraining

## Files

### Core Scripts

- `generate_dataset.py` - Synthetic dataset generator with edge cases
- `benchmark_v2.py` - Trained classifier benchmark (recommended)
- `benchmark_lang_aware.py` - Language-aware routing test
- `analyze_errors.py` - Error analysis tool
- `test_edge_cases.py` - Edge case validator

### Deprecated

- `benchmark.py` - Original reference embedding approach (poor accuracy)

### Data

- `data/routing/synthetic_with_edges.jsonl` - Generated dataset
- `data/routing/edge_cases.jsonl` - Hand-crafted edge cases
- `results/*.csv` - Benchmark outputs

## Dataset Generation

### Generate Custom Dataset

```bash
python3 tests/routing/generate_dataset.py \
    --count 2000 \
    --output data/routing/custom.jsonl \
    --edge-case-rate 0.5 \
    --typo-rate 0.15 \
    --seed 42
```

**Parameters:**
- `--count`: Total examples (default: 1000)
- `--edge-case-rate`: Probability of edge case (default: 0.3)
- `--typo-rate`: Probability of typos (default: 0.2)
- `--seed`: Random seed for reproducibility

### Edge Case Templates

The generator includes realistic edge cases:

**Control Commands:**
- Implicit: Environmental conditions (`"il fait noir"`)
- Minimal: Single words (`"lights"`, `"chauffage"`)
- Contextual: Polite requests (`"peux-tu m'aider avec les lumières"`)

**Chat Queries:**
- Status: State questions (`"are the lights on?"`)
- Info: Device inventory (`"what lights do I have?"`)

## Error Analysis

### Find Misclassifications

```bash
python3 tests/routing/analyze_errors.py
```

Shows which examples failed and why, grouped by error type:
- False Positives (chat → control) - most critical
- False Negatives (control → chat)

### Test Edge Cases

```bash
python3 tests/routing/test_edge_cases.py
```

Tests model on hand-crafted edge cases from `data/routing/edge_cases.jsonl`.

## Iterative Improvement

### 1. Collect Production Data

```bash
# Export LLM logs with routing decisions
# Label as control if tools were called, chat otherwise
```

### 2. Retrain with Real Examples

```bash
python3 tests/routing/train_classifier.py \
    --base-model all-MiniLM-L12-v2 \
    --synthetic data/routing/synthetic_with_edges.jsonl \
    --production data/routing/production_examples.jsonl \
    --output data/routing/models/aria-router-v2
```

### 3. Re-benchmark

```bash
python3 tests/routing/benchmark_v2.py \
    --models data/routing/models/aria-router-v2 \
    --dataset data/routing/production_examples.jsonl \
    --output results/v2_benchmark.csv
```

## Success Criteria

### Minimum Viable
- ✅ Binary classifier with >85% F1 score
- ✅ Inference latency P95 < 20ms on Xeon E3
- ✅ Zero crashes during 100-request load test

### Target (Achieved)
- ✅ >90% precision (97%)
- ✅ >90% recall (97%)
- ✅ Latency P50 < 15ms (12ms)
- ✅ Handles French + English

### Stretch Goals
- ⬜ Fine-tuned model outperforms zero-shot by >5% F1
- ⬜ Query caching reduces P50 to <5ms
- ⬜ Real-time retraining from production feedback

## Next Steps

1. ✅ Test embeddings for chat vs control routing
2. ✅ Compare multilingual vs language-aware approaches
3. ⬜ Implement `EmbeddingRouter` class in `app/llm/embedding_router.py`
4. ⬜ Integrate into `/api/llm/process` endpoint
5. ⬜ Deploy with conservative threshold (0.6)
6. ⬜ Monitor for 1 week and collect misclassifications
7. ⬜ Retrain with production data monthly

## References

- Sentence Transformers: https://www.sbert.net/
- all-MiniLM-L6-v2: 80MB, 384 dims, multilingual
- all-MiniLM-L12-v2: 120MB, 384 dims, better accuracy
