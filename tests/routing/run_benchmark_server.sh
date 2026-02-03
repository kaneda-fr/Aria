#!/bin/bash
# Run trilingual benchmark on server (192.168.100.100)
# This script should be executed ON THE SERVER after syncing files

set -e  # Exit on error

echo "=================================================="
echo "Aria Trilingual Intent Routing Benchmark"
echo "French + English + German"
echo "=================================================="
echo ""

# Configuration
DATASET_SIZE=1000
DATASET_PATH="data/routing/synthetic_trilingual_edges.jsonl"
ONNX_MODEL_PATH="models/all-MiniLM-L6-v2-onnx-int8"
RESULTS_PATH="results/trilingual_onnx_comparison.csv"

# Set HuggingFace cache to project directory
export HF_HOME="$(pwd)/.cache/huggingface"
export TRANSFORMERS_CACHE="$(pwd)/.cache/huggingface/transformers"

echo "Environment:"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo ""

# Create directories
mkdir -p data/routing
mkdir -p models
mkdir -p results
mkdir -p .cache/huggingface/transformers

echo "Step 1: Generate trilingual test dataset"
echo "  Size: $DATASET_SIZE examples (FR + EN + DE)"
echo "  Edge cases: 40%"
echo "  Typos: 20%"
echo ""

python3 tests/routing/generate_dataset.py \
    --count $DATASET_SIZE \
    --output $DATASET_PATH \
    --edge-case-rate 0.4 \
    --typo-rate 0.2 \
    --seed 42

echo ""
echo "Step 2: Convert model to ONNX INT8"
echo "  Model: sentence-transformers/all-MiniLM-L6-v2"
echo "  Output: $ONNX_MODEL_PATH"
echo ""

if [ -d "$ONNX_MODEL_PATH" ]; then
    echo "⚠️  ONNX model already exists, skipping conversion"
    echo "   (Delete $ONNX_MODEL_PATH to regenerate)"
else
    python3 tests/routing/convert_to_onnx.py \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --output $ONNX_MODEL_PATH
fi

echo ""
echo "Step 3: Run benchmark (PyTorch vs ONNX INT8)"
echo "  Dataset: $DATASET_PATH"
echo "  Results: $RESULTS_PATH"
echo ""

python3 tests/routing/benchmark_onnx.py \
    --pytorch-model sentence-transformers/all-MiniLM-L6-v2 \
    --onnx-model $ONNX_MODEL_PATH \
    --dataset $DATASET_PATH \
    --output $RESULTS_PATH

echo ""
echo "=================================================="
echo "✓ Benchmark complete!"
echo ""
echo "Results saved to: $RESULTS_PATH"
echo ""
echo "To view results:"
echo "  cat $RESULTS_PATH"
echo ""
echo "To deploy in production:"
echo "  export ARIA_EMBEDDING_ROUTER_ENABLED=1"
echo "  export ARIA_EMBEDDING_ROUTER_MODEL=$ONNX_MODEL_PATH"
echo "  export ARIA_EMBEDDING_ROUTER_THRESHOLD=0.5"
echo "  export HF_HOME=\$(pwd)/.cache/huggingface"
echo "  export TRANSFORMERS_CACHE=\$(pwd)/.cache/huggingface/transformers"
echo "=================================================="
