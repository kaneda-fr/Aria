#!/usr/bin/env python3
"""
Benchmark ONNX quantized models for intent routing.

Compares regular sentence-transformers vs ONNX quantized versions.
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
except ImportError as e:
    print(f"Error: Missing required dependencies.")
    print(f"Install with: pip install sentence-transformers scikit-learn")
    raise e


@dataclass
class BenchmarkResult:
    """Results for a model configuration."""
    model_name: str
    model_type: str  # "pytorch" or "onnx-int8"
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    accuracy: float


class ONNXIntentClassifier:
    """Classifier using ONNX Runtime for faster inference."""

    def __init__(self, model_path: str):
        """Load ONNX model with optimized runtime."""
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "ONNX support requires: pip install optimum[onnxruntime] transformers"
            )

        print(f"Loading ONNX model: {model_path}...", end=" ", flush=True)
        start = time.time()

        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_path,
            provider="CPUExecutionProvider"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.classifier = None
        self.model_name = Path(model_path).name

        load_time = time.time() - start
        print(f"✓ ({load_time:.1f}s)")

    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """Encode texts to embeddings using ONNX runtime."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Run ONNX inference
        outputs = self.model(**inputs)

        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings

    def train(self, texts: List[str], labels: List[str]):
        """Train logistic regression on embeddings."""
        print("Generating embeddings for training...", end=" ", flush=True)
        embeddings = self.encode(texts)
        print("✓")

        y = np.array([1 if label == "control" else 0 for label in labels])

        print("Training logistic regression...", end=" ", flush=True)
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.classifier.fit(embeddings, y)
        print("✓")

    def predict_single(self, text: str) -> Tuple[str, float, float]:
        """Predict single example with latency measurement."""
        start = time.perf_counter()

        embedding = self.encode([text])[0]
        prob = self.classifier.predict_proba([embedding])[0]
        confidence = prob[1]
        prediction = "control" if confidence >= 0.5 else "chat"

        latency_ms = (time.perf_counter() - start) * 1000

        return prediction, confidence, latency_ms

    def predict(self, examples: List[Dict]) -> Tuple[List[str], List[float], List[float]]:
        """Predict on test examples."""
        texts = [ex["text"] for ex in examples]

        # Batch encode
        embeddings = self.encode(texts)
        probs = self.classifier.predict_proba(embeddings)
        confidences = probs[:, 1]
        predictions = ["control" if c >= 0.5 else "chat" for c in confidences]

        # Measure per-example latency on sample
        print("  Measuring per-example latency...", end=" ", flush=True)
        latencies = []
        sample_size = min(100, len(examples))
        sample_indices = np.random.choice(len(examples), sample_size, replace=False)

        for idx in sample_indices:
            _, _, latency = self.predict_single(examples[idx]["text"])
            latencies.append(latency)
        print("✓")

        return list(predictions), list(confidences), latencies


class PyTorchIntentClassifier:
    """Standard sentence-transformers classifier (for comparison)."""

    def __init__(self, model_name: str):
        """Load sentence transformer model."""
        print(f"Loading PyTorch model: {model_name}...", end=" ", flush=True)
        start = time.time()
        self.model = SentenceTransformer(model_name)
        load_time = time.time() - start
        print(f"✓ ({load_time:.1f}s)")

        self.model_name = model_name
        self.classifier = None

    def train(self, texts: List[str], labels: List[str]):
        """Train logistic regression on embeddings."""
        print("Generating embeddings for training...", end=" ", flush=True)
        embeddings = self.model.encode(texts, show_progress_bar=False)
        print("✓")

        y = np.array([1 if label == "control" else 0 for label in labels])

        print("Training logistic regression...", end=" ", flush=True)
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.classifier.fit(embeddings, y)
        print("✓")

    def predict_single(self, text: str) -> Tuple[str, float, float]:
        """Predict single example with latency measurement."""
        start = time.perf_counter()

        embedding = self.model.encode([text], show_progress_bar=False)[0]
        prob = self.classifier.predict_proba([embedding])[0]
        confidence = prob[1]
        prediction = "control" if confidence >= 0.5 else "chat"

        latency_ms = (time.perf_counter() - start) * 1000

        return prediction, confidence, latency_ms

    def predict(self, examples: List[Dict]) -> Tuple[List[str], List[float], List[float]]:
        """Predict on test examples."""
        texts = [ex["text"] for ex in examples]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        probs = self.classifier.predict_proba(embeddings)
        confidences = probs[:, 1]
        predictions = ["control" if c >= 0.5 else "chat" for c in confidences]

        # Measure per-example latency
        print("  Measuring per-example latency...", end=" ", flush=True)
        latencies = []
        sample_size = min(100, len(examples))
        sample_indices = np.random.choice(len(examples), sample_size, replace=False)

        for idx in sample_indices:
            _, _, latency = self.predict_single(examples[idx]["text"])
            latencies.append(latency)
        print("✓")

        return list(predictions), list(confidences), latencies


def load_dataset(path: Path) -> List[Dict]:
    """Load dataset from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def evaluate(classifier, test_examples: List[Dict], model_type: str) -> BenchmarkResult:
    """Evaluate classifier on test set."""
    labels = [ex["label"] for ex in test_examples]

    print("Evaluating on test set...")
    predictions, confidences, latencies = classifier.predict(test_examples)

    # Compute metrics
    y_true = np.array([1 if label == "control" else 0 for label in labels])
    y_pred = np.array([1 if pred == "control" else 0 for pred in predictions])

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    accuracy = (y_true == y_pred).mean()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Latency stats
    latencies_np = np.array(latencies)
    latency_p50 = np.percentile(latencies_np, 50)
    latency_p95 = np.percentile(latencies_np, 95)

    return BenchmarkResult(
        model_name=classifier.model_name,
        model_type=model_type,
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_positive_rate=fp_rate,
        false_negative_rate=fn_rate,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        accuracy=accuracy,
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print comparison table."""
    print("\n" + "=" * 140)
    print("PyTorch vs ONNX-INT8 Comparison")
    print("=" * 140)

    header = (
        f"{'Model':<35} {'Type':<12} {'Acc':>5} {'Prec':>6} {'Recall':>7} {'F1':>6} "
        f"{'FP Rate':>8} {'FN Rate':>8} {'P50 (ms)':>9} {'P95 (ms)':>9}"
    )
    print(header)
    print("-" * 140)

    for result in results:
        row = (
            f"{result.model_name:<35} "
            f"{result.model_type:<12} "
            f"{result.accuracy:>5.3f} "
            f"{result.precision:>6.3f} "
            f"{result.recall:>7.3f} "
            f"{result.f1_score:>6.3f} "
            f"{result.false_positive_rate:>8.3f} "
            f"{result.false_negative_rate:>8.3f} "
            f"{result.latency_p50_ms:>9.1f} "
            f"{result.latency_p95_ms:>9.1f}"
        )
        print(row)

    print("=" * 140)

    # Calculate speedup
    pytorch_results = [r for r in results if r.model_type == "pytorch"]
    onnx_results = [r for r in results if r.model_type == "onnx-int8"]

    if pytorch_results and onnx_results:
        pt = pytorch_results[0]
        ox = onnx_results[0]

        speedup = (pt.latency_p50_ms / ox.latency_p50_ms - 1) * 100
        accuracy_delta = (ox.accuracy - pt.accuracy) * 100

        print(f"\n⚡ ONNX INT8 Speedup:")
        print(f"  Latency: {pt.latency_p50_ms:.1f}ms → {ox.latency_p50_ms:.1f}ms ({speedup:+.1f}%)")
        print(f"  Accuracy: {pt.accuracy:.3f} → {ox.accuracy:.3f} ({accuracy_delta:+.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX vs PyTorch models")
    parser.add_argument(
        "--pytorch-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="PyTorch model name"
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path("models/all-MiniLM-L6-v2-onnx-int8"),
        help="Path to ONNX model directory"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to JSONL dataset file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/onnx_benchmark.csv"),
        help="Output CSV file path"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    examples = load_dataset(args.dataset)
    print(f"✓ Loaded {len(examples)} examples")

    # Split train/test
    train_examples, test_examples = train_test_split(
        examples, test_size=args.test_split, random_state=42,
        stratify=[e["label"] for e in examples]
    )
    print(f"  - Training: {len(train_examples)}")
    print(f"  - Testing:  {len(test_examples)}")

    train_texts = [ex["text"] for ex in train_examples]
    train_labels = [ex["label"] for ex in train_examples]

    all_results = []

    # Test PyTorch model
    print(f"\n{'=' * 80}")
    print(f"[1/2] PyTorch Model: {args.pytorch_model}")
    print(f"{'=' * 80}")
    pytorch_classifier = PyTorchIntentClassifier(args.pytorch_model)
    pytorch_classifier.train(train_texts, train_labels)
    result_pytorch = evaluate(pytorch_classifier, test_examples, "pytorch")
    all_results.append(result_pytorch)

    # Test ONNX model
    if args.onnx_model.exists():
        print(f"\n{'=' * 80}")
        print(f"[2/2] ONNX Model: {args.onnx_model}")
        print(f"{'=' * 80}")
        onnx_classifier = ONNXIntentClassifier(str(args.onnx_model))
        onnx_classifier.train(train_texts, train_labels)
        result_onnx = evaluate(onnx_classifier, test_examples, "onnx-int8")
        all_results.append(result_onnx)
    else:
        print(f"\n⚠️  ONNX model not found at {args.onnx_model}")
        print(f"   Run: python tests/routing/convert_to_onnx.py --model {args.pytorch_model}")

    # Print comparison
    print_results_table(all_results)

    # Save CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name", "model_type", "accuracy", "precision", "recall", "f1_score",
            "false_positive_rate", "false_negative_rate", "latency_p50_ms", "latency_p95_ms"
        ])
        for result in all_results:
            writer.writerow([
                result.model_name, result.model_type, f"{result.accuracy:.4f}",
                f"{result.precision:.4f}", f"{result.recall:.4f}", f"{result.f1_score:.4f}",
                f"{result.false_positive_rate:.4f}", f"{result.false_negative_rate:.4f}",
                f"{result.latency_p50_ms:.2f}", f"{result.latency_p95_ms:.2f}"
            ])

    print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
