#!/usr/bin/env python3
"""
Improved benchmark for embedding-based intent classification.

Uses trained logistic regression on embeddings instead of reference embeddings.
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
    """Results for a single model."""
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
    accuracy: float
    confusion_mat: List[List[int]]


class TrainedIntentClassifier:
    """Embedding + logistic regression classifier."""

    def __init__(self, model_name: str):
        """Load sentence transformer model."""
        print(f"Loading embedding model: {model_name}...", end=" ", flush=True)
        start = time.time()
        self.model = SentenceTransformer(model_name)
        load_time = time.time() - start
        print(f"✓ ({load_time:.1f}s)")

        self.model_name = model_name
        self.classifier = None

    def train(self, texts: List[str], labels: List[str]):
        """Train classifier on embeddings."""
        print("Generating embeddings for training...", end=" ", flush=True)
        embeddings = self.model.encode(texts, show_progress_bar=False)
        print("✓")

        # Convert labels to binary
        y = np.array([1 if label == "control" else 0 for label in labels])

        # Train logistic regression
        print("Training logistic regression...", end=" ", flush=True)
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.classifier.fit(embeddings, y)
        print("✓")

    def predict_single(self, text: str) -> Tuple[str, float, float]:
        """
        Predict single example (for latency measurement).

        Returns:
            (prediction, confidence, latency_ms)
        """
        start = time.perf_counter()

        # Generate embedding
        embedding = self.model.encode([text], show_progress_bar=False)[0]

        # Get probability from classifier
        prob = self.classifier.predict_proba([embedding])[0]
        confidence = prob[1]  # Probability of "control"

        # Predict
        prediction = "control" if confidence >= 0.5 else "chat"

        latency_ms = (time.perf_counter() - start) * 1000

        return prediction, confidence, latency_ms

    def predict(self, texts: List[str]) -> Tuple[List[str], List[float], List[float]]:
        """
        Predict intent for texts (batch for accuracy, individual for latency).

        Returns:
            (predictions, confidences, latencies_ms)
        """
        # Batch embeddings for efficiency
        embeddings = self.model.encode(texts, show_progress_bar=False)
        probs = self.classifier.predict_proba(embeddings)
        confidences = probs[:, 1]  # Probability of "control"
        predictions = ["control" if c >= 0.5 else "chat" for c in confidences]

        # Measure per-example latency by re-encoding individually
        print("  Measuring per-example latency...", end=" ", flush=True)
        latencies = []
        # Sample 100 examples for latency measurement to save time
        sample_size = min(100, len(texts))
        sample_indices = np.random.choice(len(texts), sample_size, replace=False)

        for idx in sample_indices:
            _, _, latency = self.predict_single(texts[idx])
            latencies.append(latency)
        print("✓")

        return predictions, confidences.tolist(), latencies


def load_dataset(path: Path) -> List[Dict]:
    """Load dataset from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def evaluate_model(
    classifier: TrainedIntentClassifier,
    test_examples: List[Dict]
) -> BenchmarkResult:
    """Evaluate model on test set."""
    texts = [ex["text"] for ex in test_examples]
    labels = [ex["label"] for ex in test_examples]

    # Get predictions
    predictions, confidences, latencies = classifier.predict(texts)

    # Compute metrics
    y_true = np.array([1 if label == "control" else 0 for label in labels])
    y_pred = np.array([1 if pred == "control" else 0 for pred in predictions])

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    accuracy = (y_true == y_pred).mean()

    # Confusion matrix for FP/FN rates
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # False positive rate: chat → control
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # False negative rate: control → chat
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Latency stats
    latencies_np = np.array(latencies)
    latency_p50 = np.percentile(latencies_np, 50)
    latency_p95 = np.percentile(latencies_np, 95)
    latency_p99 = np.percentile(latencies_np, 99)

    return BenchmarkResult(
        model_name=classifier.model_name,
        threshold=0.5,  # Fixed for logistic regression
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_positive_rate=fp_rate,
        false_negative_rate=fn_rate,
        latency_p50_ms=latency_p50,
        latency_p95_ms=latency_p95,
        latency_p99_ms=latency_p99,
        accuracy=accuracy,
        confusion_mat=cm.tolist(),
    )


def print_confusion_matrix(result: BenchmarkResult):
    """Print confusion matrix."""
    cm = np.array(result.confusion_mat)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                Chat    Control")
    print(f"Actual  Chat     {tn:4d}    {fp:4d}")
    print(f"        Control  {fn:4d}    {tp:4d}")
    print(f"\nMetrics:")
    print(f"  True Negatives:  {tn} (chat correctly identified)")
    print(f"  False Positives: {fp} (chat misclassified as control) ⚠️")
    print(f"  False Negatives: {fn} (control misclassified as chat)")
    print(f"  True Positives:  {tp} (control correctly identified)")


def print_results_table(results: List[BenchmarkResult]):
    """Print results as formatted table."""
    print("\n" + "=" * 140)
    print("Model Comparison - Embedding-Based Intent Routing (Trained Classifier)")
    print("=" * 140)

    # Header
    header = (
        f"{'Model':<35} {'Acc':>5} {'Prec':>6} {'Recall':>7} {'F1':>6} "
        f"{'FP Rate':>8} {'FN Rate':>8} {'P50 (ms)':>9} {'P95 (ms)':>9}"
    )
    print(header)
    print("-" * 140)

    # Rows
    for result in results:
        row = (
            f"{result.model_name:<35} "
            f"{result.accuracy:>5.2f} "
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
    print("  Acc:     Overall accuracy")
    print("  Prec:    Precision (% of control predictions that are correct)")
    print("  Recall:  Recall (% of control examples caught)")
    print("  F1:      Harmonic mean of precision and recall")
    print("  FP Rate: False positive rate (chat → control mistakes) ⚠️ CRITICAL")
    print("  FN Rate: False negative rate (control → chat mistakes)")
    print("  P50/P95: Latency percentiles in milliseconds (includes embedding generation)")

    # Recommendation
    print("\n⚠️  Target Metrics for Production:")
    print("  - Precision: >0.90 (minimize false device activations)")
    print("  - Recall: >0.85 (catch most control commands)")
    print("  - FP Rate: <0.05 (less than 5% chat → control errors)")
    print("  - Latency P95: <20ms on Xeon E3")


def save_results_csv(results: List[BenchmarkResult], output_path: Path):
    """Save results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "model_name",
            "accuracy",
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
                f"{result.accuracy:.4f}",
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
    parser = argparse.ArgumentParser(description="Benchmark embedding models with trained classifier")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model names"
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
        default=Path("results/benchmark_v2.csv"),
        help="Output CSV file path"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    examples = load_dataset(args.dataset)
    print(f"✓ Loaded {len(examples)} examples")

    # Split train/test
    train_examples, test_examples = train_test_split(
        examples, test_size=args.test_split, random_state=42, stratify=[e["label"] for e in examples]
    )
    print(f"  - Training: {len(train_examples)} ({sum(1 for e in train_examples if e['label'] == 'control')} control)")
    print(f"  - Testing:  {len(test_examples)} ({sum(1 for e in test_examples if e['label'] == 'control')} control)")

    # Parse model names
    model_names = [m.strip() for m in args.models.split(",")]
    print(f"\nBenchmarking {len(model_names)} model(s)...")

    # Evaluate each model
    all_results = []

    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")

        # Create and train classifier
        classifier = TrainedIntentClassifier(model_name)
        train_texts = [ex["text"] for ex in train_examples]
        train_labels = [ex["label"] for ex in train_examples]
        classifier.train(train_texts, train_labels)

        # Evaluate on test set
        print("Evaluating on test set...")
        result = evaluate_model(classifier, test_examples)
        all_results.append(result)

        print(f"\nResults:")
        print(f"  Accuracy:     {result.accuracy:.3f}")
        print(f"  Precision:    {result.precision:.3f}")
        print(f"  Recall:       {result.recall:.3f}")
        print(f"  F1 Score:     {result.f1_score:.3f}")
        print(f"  FP Rate:      {result.false_positive_rate:.3f} {'⚠️ TOO HIGH' if result.false_positive_rate > 0.10 else '✓'}")
        print(f"  FN Rate:      {result.false_negative_rate:.3f}")
        print(f"  Latency P50:  {result.latency_p50_ms:.1f}ms")
        print(f"  Latency P95:  {result.latency_p95_ms:.1f}ms")

        print_confusion_matrix(result)

    # Print comparison table
    if len(all_results) > 1:
        print_results_table(all_results)

    # Save CSV
    save_results_csv(all_results, args.output)


if __name__ == "__main__":
    main()
