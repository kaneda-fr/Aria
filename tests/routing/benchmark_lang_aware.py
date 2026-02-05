#!/usr/bin/env python3
"""
Language-aware benchmark: Train separate classifiers per language.

Tests if language-specific routing improves accuracy vs single multilingual classifier.
"""

import argparse
import csv
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

import json
import time


@dataclass
class BenchmarkResult:
    """Results for a model configuration."""
    model_name: str
    approach: str  # "multilingual" or "language-aware"
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    accuracy: float


class LanguageAwareClassifier:
    """Router with separate classifiers per language."""

    def __init__(self, model_name: str):
        """Load sentence transformer model."""
        print(f"Loading embedding model: {model_name}...", end=" ", flush=True)
        start = time.time()
        self.model = SentenceTransformer(model_name)
        load_time = time.time() - start
        print(f"✓ ({load_time:.1f}s)")

        self.model_name = model_name
        self.classifiers = {}  # lang -> LogisticRegression

    def train(self, examples: List[Dict]):
        """Train separate classifiers for each language."""
        # Split by language
        fr_examples = [ex for ex in examples if ex["lang"] == "fr"]
        en_examples = [ex for ex in examples if ex["lang"] == "en"]

        print(f"Training language-specific classifiers...")
        print(f"  French: {len(fr_examples)} examples")
        print(f"  English: {len(en_examples)} examples")

        # Train French classifier
        if fr_examples:
            print("  Training French classifier...", end=" ", flush=True)
            fr_texts = [ex["text"] for ex in fr_examples]
            fr_labels = [1 if ex["label"] == "control" else 0 for ex in fr_examples]
            fr_embeddings = self.model.encode(fr_texts, show_progress_bar=False)

            self.classifiers["fr"] = LogisticRegression(max_iter=1000, random_state=42)
            self.classifiers["fr"].fit(fr_embeddings, fr_labels)
            print("✓")

        # Train English classifier
        if en_examples:
            print("  Training English classifier...", end=" ", flush=True)
            en_texts = [ex["text"] for ex in en_examples]
            en_labels = [1 if ex["label"] == "control" else 0 for ex in en_examples]
            en_embeddings = self.model.encode(en_texts, show_progress_bar=False)

            self.classifiers["en"] = LogisticRegression(max_iter=1000, random_state=42)
            self.classifiers["en"].fit(en_embeddings, en_labels)
            print("✓")

    def predict_single(self, text: str, lang: str) -> Tuple[str, float, float]:
        """
        Predict with language-specific classifier.

        Args:
            text: Input text
            lang: Detected language ("fr" or "en")

        Returns:
            (prediction, confidence, latency_ms)
        """
        start = time.perf_counter()

        # Get appropriate classifier
        classifier = self.classifiers.get(lang)
        if classifier is None:
            # Fallback to first available if language not found
            classifier = list(self.classifiers.values())[0]

        # Generate embedding
        embedding = self.model.encode([text], show_progress_bar=False)[0]

        # Get probability
        prob = classifier.predict_proba([embedding])[0]
        confidence = prob[1]  # Probability of "control"

        # Predict
        prediction = "control" if confidence >= 0.5 else "chat"

        latency_ms = (time.perf_counter() - start) * 1000

        return prediction, confidence, latency_ms

    def predict(self, examples: List[Dict]) -> Tuple[List[str], List[float], List[float]]:
        """
        Predict on test examples using language-aware routing.

        Returns:
            (predictions, confidences, latencies_ms)
        """
        predictions = []
        confidences = []

        # Group by language for batch processing
        lang_groups = {}
        for i, ex in enumerate(examples):
            lang = ex["lang"]
            if lang not in lang_groups:
                lang_groups[lang] = []
            lang_groups[lang].append((i, ex))

        # Process each language group
        results = [None] * len(examples)  # Maintain order
        for lang, group in lang_groups.items():
            indices, exs = zip(*group)
            texts = [ex["text"] for ex in exs]

            # Get classifier
            classifier = self.classifiers.get(lang, list(self.classifiers.values())[0])

            # Batch encode
            embeddings = self.model.encode(texts, show_progress_bar=False)
            probs = classifier.predict_proba(embeddings)
            confs = probs[:, 1]  # Probability of "control"
            preds = ["control" if c >= 0.5 else "chat" for c in confs]

            # Store results in original order
            for idx, pred, conf in zip(indices, preds, confs):
                results[idx] = (pred, conf)

        predictions, confidences = zip(*results)

        # Measure per-example latency on sample
        print("  Measuring per-example latency...", end=" ", flush=True)
        latencies = []
        sample_size = min(100, len(examples))
        sample_indices = np.random.choice(len(examples), sample_size, replace=False)

        for idx in sample_indices:
            ex = examples[idx]
            _, _, latency = self.predict_single(ex["text"], ex["lang"])
            latencies.append(latency)
        print("✓")

        return list(predictions), list(confidences), latencies


class MultilingualClassifier:
    """Single classifier for all languages (baseline)."""

    def __init__(self, model_name: str):
        """Load sentence transformer model."""
        print(f"Loading embedding model: {model_name}...", end=" ", flush=True)
        start = time.time()
        self.model = SentenceTransformer(model_name)
        load_time = time.time() - start
        print(f"✓ ({load_time:.1f}s)")

        self.model_name = model_name
        self.classifier = None

    def train(self, examples: List[Dict]):
        """Train single classifier on all languages."""
        print(f"Training multilingual classifier...")
        texts = [ex["text"] for ex in examples]
        labels = [1 if ex["label"] == "control" else 0 for ex in examples]

        embeddings = self.model.encode(texts, show_progress_bar=False)

        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.classifier.fit(embeddings, labels)
        print("✓")

    def predict_single(self, text: str, lang: str = None) -> Tuple[str, float, float]:
        """Predict (lang parameter ignored for compatibility)."""
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


def load_dataset(path: Path) -> List[Dict]:
    """Load dataset from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def evaluate(classifier, test_examples: List[Dict], approach: str) -> BenchmarkResult:
    """Evaluate classifier on test set."""
    labels = [ex["label"] for ex in test_examples]

    # Get predictions
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
        approach=approach,
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
    print("Language-Aware vs Multilingual Routing Comparison")
    print("=" * 140)

    header = (
        f"{'Model':<30} {'Approach':<18} {'Acc':>5} {'Prec':>6} {'Recall':>7} {'F1':>6} "
        f"{'FP Rate':>8} {'FN Rate':>8} {'P50 (ms)':>9} {'P95 (ms)':>9}"
    )
    print(header)
    print("-" * 140)

    for result in results:
        row = (
            f"{result.model_name:<30} "
            f"{result.approach:<18} "
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

    # Find best per approach
    multi = [r for r in results if r.approach == "multilingual"]
    lang_aware = [r for r in results if r.approach == "language-aware"]

    if multi and lang_aware:
        best_multi = max(multi, key=lambda r: r.f1_score)
        best_lang = max(lang_aware, key=lambda r: r.f1_score)

        print("\n📊 Comparison:")
        print(f"  Multilingual:    F1={best_multi.f1_score:.3f}, FP={best_multi.false_positive_rate:.3f}, Latency={best_multi.latency_p50_ms:.1f}ms")
        print(f"  Language-Aware:  F1={best_lang.f1_score:.3f}, FP={best_lang.false_positive_rate:.3f}, Latency={best_lang.latency_p50_ms:.1f}ms")

        improvement = (best_lang.f1_score - best_multi.f1_score) / best_multi.f1_score * 100
        if improvement > 0.5:
            print(f"\n✅ Language-aware routing improves F1 by {improvement:.1f}%")
        elif improvement < -0.5:
            print(f"\n⚠️  Multilingual performs better by {abs(improvement):.1f}%")
        else:
            print(f"\n➡️  No significant difference ({improvement:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Compare language-aware vs multilingual routing")
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
        default=Path("results/lang_aware_benchmark.csv"),
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
    print(f"  - Training: {len(train_examples)}")
    print(f"  - Testing:  {len(test_examples)}")

    # Parse model names
    model_names = [m.strip() for m in args.models.split(",")]

    all_results = []

    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")

        # Test multilingual approach
        print("\n[1/2] Multilingual Approach")
        print("-" * 40)
        multi = MultilingualClassifier(model_name)
        multi.train(train_examples)
        print("Evaluating on test set...")
        result_multi = evaluate(multi, test_examples, "multilingual")
        all_results.append(result_multi)

        print(f"  Accuracy: {result_multi.accuracy:.3f}, F1: {result_multi.f1_score:.3f}, FP Rate: {result_multi.false_positive_rate:.3f}")

        # Test language-aware approach
        print("\n[2/2] Language-Aware Approach")
        print("-" * 40)
        lang_aware = LanguageAwareClassifier(model_name)
        lang_aware.train(train_examples)
        print("Evaluating on test set...")
        result_lang = evaluate(lang_aware, test_examples, "language-aware")
        all_results.append(result_lang)

        print(f"  Accuracy: {result_lang.accuracy:.3f}, F1: {result_lang.f1_score:.3f}, FP Rate: {result_lang.false_positive_rate:.3f}")

    # Print comparison
    print_results_table(all_results)

    # Save CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name", "approach", "accuracy", "precision", "recall", "f1_score",
            "false_positive_rate", "false_negative_rate", "latency_p50_ms", "latency_p95_ms"
        ])
        for result in all_results:
            writer.writerow([
                result.model_name, result.approach, f"{result.accuracy:.4f}",
                f"{result.precision:.4f}", f"{result.recall:.4f}", f"{result.f1_score:.4f}",
                f"{result.false_positive_rate:.4f}", f"{result.false_negative_rate:.4f}",
                f"{result.latency_p50_ms:.2f}", f"{result.latency_p95_ms:.2f}"
            ])

    print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
