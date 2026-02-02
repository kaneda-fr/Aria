#!/usr/bin/env python3
"""
Benchmark embedding models for chat vs control intent classification.

Evaluates multiple models on accuracy and latency metrics.
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
except ImportError as e:
    print(f"Error: Missing required dependencies.")
    print(f"Install with: pip install sentence-transformers scikit-learn")
    raise e


@dataclass
class BenchmarkResult:
    """Results for a single model at a specific threshold."""
    model_name: str
    threshold: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    model_size_mb: float


class IntentClassifier:
    """Simple embedding-based binary classifier."""

    def __init__(self, model_name: str):
        """Load sentence transformer model."""
        print(f"Loading model: {model_name}...", end=" ", flush=True)
        start = time.time()
        self.model = SentenceTransformer(model_name)
        load_time = time.time() - start
        print(f"✓ ({load_time:.1f}s)")

        self.model_name = model_name

        # Pre-compute reference embeddings for control and chat
        control_refs = [
            "turn on the light",
            "allume la lumière",
            "open the shutter",
            "ouvre le volet",
            "set brightness to 50",
            "éteins les lumières",
        ]
        chat_refs = [
            "what time is it?",
            "quelle heure est-il ?",
            "tell me a joke",
            "raconte-moi une blague",
            "how are you?",
            "comment vas-tu ?",
        ]

        print("Computing reference embeddings...", end=" ", flush=True)
        self.control_embedding = self.model.encode(control_refs).mean(axis=0)
        self.chat_embedding = self.model.encode(chat_refs).mean(axis=0)
        print("✓")

    def predict(self, texts: List[str], threshold: float = 0.5) -> Tuple[List[str], List[float], List[float]]:
        """
        Predict intent for texts.

        Returns:
            (predictions, confidences, latencies_ms)
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)

        predictions = []
        confidences = []
        latencies = []

        for emb in embeddings:
            start = time.perf_counter()

            # Cosine similarity to reference embeddings
            control_sim = np.dot(emb, self.control_embedding) / (
                np.linalg.norm(emb) * np.linalg.norm(self.control_embedding)
            )
            chat_sim = np.dot(emb, self.chat_embedding) / (
                np.linalg.norm(emb) * np.linalg.norm(self.chat_embedding)
            )

            # Normalize to [0, 1] range
            # Higher control_sim → more likely control
            confidence = (control_sim + 1) / 2  # cosine sim is in [-1, 1]

            if confidence >= threshold:
                intent = "control"
            else:
                intent = "chat"

            latency_ms = (time.perf_counter() - start) * 1000

            predictions.append(intent)
            confidences.append(confidence)
            latencies.append(latency_ms)

        return predictions, confidences, latencies


def load_dataset(path: Path) -> List[Dict]:
    """Load dataset from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def find_optimal_threshold(
    labels: List[str],
    confidences: List[float],
    thresholds: List[float]
) -> Tuple[float, float]:
    """
    Find threshold that maximizes F1 score.

    Returns:
        (best_threshold, best_f1)
    """
    best_f1 = 0.0
    best_threshold = 0.5

    for thresh in thresholds:
        predictions = ["control" if c >= thresh else "chat" for c in confidences]

        # Compute F1
        y_true = [1 if label == "control" else 0 for label in labels]
        y_pred = [1 if pred == "control" else 0 for pred in predictions]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold, best_f1


def evaluate_model(
    classifier: IntentClassifier,
    examples: List[Dict],
    threshold: float
) -> BenchmarkResult:
    """Evaluate model at given threshold."""
    texts = [ex["text"] for ex in examples]
    labels = [ex["label"] for ex in examples]

    # Get predictions
    predictions, confidences, latencies = classifier.predict(texts, threshold)

    # Compute metrics
    y_true = [1 if label == "control" else 0 for label in labels]
    y_pred = [1 if pred == "control" else 0 for pred in predictions]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    # Confusion matrix for FP/FN rates
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # False positive rate: chat → control
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # False negative rate: control → chat
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Latency stats (exclude first prediction due to warmup)
    latencies_np = np.array(latencies[1:]) if len(latencies) > 1 else np.array(latencies)
    latency_p50 = np.percentile(latencies_np, 50)
    latency_p95 = np.percentile(latencies_np, 95)
    latency_p99 = np.percentile(latencies_np, 99)

    # Model size (approximate)
    model_size_mb = 0  # TODO: compute actual size

    return BenchmarkResult(
        model_name=classifier.model_name,
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_positive_rate=fp_rate,
        false_negative_rate=fn_rate,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        latency_p99_ms=latency_p99,
        model_size_mb=model_size_mb,
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print results as formatted table."""
    print("\n" + "=" * 140)
    print("Model Comparison - Embedding-Based Intent Routing")
    print("=" * 140)

    # Header
    header = (
        f"{'Model':<35} {'Thresh':>7} {'Prec':>6} {'Recall':>7} {'F1':>6} "
        f"{'FP Rate':>8} {'FN Rate':>8} {'P50 (ms)':>9} {'P95 (ms)':>9}"
    )
    print(header)
    print("-" * 140)

    # Rows
    for result in results:
        row = (
            f"{result.model_name:<35} "
            f"{result.threshold:>7.2f} "
            f"{result.precision:>6.2f} "
            f"{result.recall:>7.2f} "
            f"{result.f1_score:>6.2f} "
            f"{result.false_positive_rate:>8.2f} "
            f"{result.false_negative_rate:>8.2f} "
            f"{result.latency_p50_ms:>9.1f} "
            f"{result.latency_p95_ms:>9.1f}"
        )
        print(row)

    print("=" * 140)

    # Legend
    print("\nLegend:")
    print("  Thresh:  Optimal classification threshold (0.0-1.0)")
    print("  Prec:    Precision (% of control predictions that are correct)")
    print("  Recall:  Recall (% of control examples caught)")
    print("  F1:      Harmonic mean of precision and recall")
    print("  FP Rate: False positive rate (chat → control mistakes)")
    print("  FN Rate: False negative rate (control → chat mistakes)")
    print("  P50/P95: Latency percentiles in milliseconds")

    # Recommendation
    print("\nRecommendation:")
    best_f1 = max(results, key=lambda r: r.f1_score)
    best_latency = min(results, key=lambda r: r.latency_p50_ms)

    if best_f1.model_name == best_latency.model_name:
        print(f"  → {best_f1.model_name} @ threshold {best_f1.threshold:.2f}")
        print(f"    Best overall: F1={best_f1.f1_score:.2f}, P50={best_f1.latency_p50_ms:.1f}ms")
    else:
        print(f"  → For accuracy: {best_f1.model_name} @ threshold {best_f1.threshold:.2f}")
        print(f"    (F1={best_f1.f1_score:.2f}, P50={best_f1.latency_p50_ms:.1f}ms)")
        print(f"  → For latency: {best_latency.model_name} @ threshold {best_latency.threshold:.2f}")
        print(f"    (F1={best_latency.f1_score:.2f}, P50={best_latency.latency_p50_ms:.1f}ms)")


def save_results_csv(results: List[BenchmarkResult], output_path: Path):
    """Save results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "model_name",
            "threshold",
            "precision",
            "recall",
            "f1_score",
            "false_positive_rate",
            "false_negative_rate",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
        ])

        # Rows
        for result in results:
            writer.writerow([
                result.model_name,
                f"{result.threshold:.3f}",
                f"{result.precision:.4f}",
                f"{result.recall:.4f}",
                f"{result.f1_score:.4f}",
                f"{result.false_positive_rate:.4f}",
                f"{result.false_negative_rate:.4f}",
                f"{result.latency_p50_ms:.2f}",
                f"{result.latency_p95_ms:.2f}",
                f"{result.latency_p99_ms:.2f}",
            ])

    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models for intent classification")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model names (e.g., 'all-MiniLM-L6-v2,all-MiniLM-L12-v2')"
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
        default=Path("results/benchmark.csv"),
        help="Output CSV file path (default: results/benchmark.csv)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold to use (default: auto-tune for each model)"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    examples = load_dataset(args.dataset)
    print(f"✓ Loaded {len(examples)} examples")
    print(f"  - Control: {sum(1 for e in examples if e['label'] == 'control')}")
    print(f"  - Chat: {sum(1 for e in examples if e['label'] == 'chat')}")

    # Parse model names
    model_names = [m.strip() for m in args.models.split(",")]
    print(f"\nBenchmarking {len(model_names)} model(s)...")

    # Evaluate each model
    all_results = []

    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")

        # Load classifier
        classifier = IntentClassifier(model_name)

        # Get predictions for threshold tuning
        texts = [ex["text"] for ex in examples]
        labels = [ex["label"] for ex in examples]
        _, confidences, _ = classifier.predict(texts, threshold=0.5)

        # Find optimal threshold or use fixed threshold
        if args.threshold is not None:
            threshold = args.threshold
            print(f"Using fixed threshold: {threshold}")
        else:
            thresholds = np.arange(0.3, 0.8, 0.05)
            threshold, best_f1 = find_optimal_threshold(labels, confidences, thresholds)
            print(f"Optimal threshold: {threshold:.2f} (F1: {best_f1:.3f})")

        # Evaluate at optimal threshold
        result = evaluate_model(classifier, examples, threshold)
        all_results.append(result)

        print(f"Results:")
        print(f"  Precision:    {result.precision:.3f}")
        print(f"  Recall:       {result.recall:.3f}")
        print(f"  F1 Score:     {result.f1_score:.3f}")
        print(f"  FP Rate:      {result.false_positive_rate:.3f}")
        print(f"  FN Rate:      {result.false_negative_rate:.3f}")
        print(f"  Latency P50:  {result.latency_p50_ms:.1f}ms")
        print(f"  Latency P95:  {result.latency_p95_ms:.1f}ms")

    # Print comparison table
    if len(all_results) > 1:
        print_results_table(all_results)

    # Save CSV
    save_results_csv(all_results, args.output)


if __name__ == "__main__":
    main()
