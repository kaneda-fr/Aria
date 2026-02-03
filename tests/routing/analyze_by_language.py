#!/usr/bin/env python3
"""
Analyze benchmark results by language and route type.

Shows detailed breakdown of accuracy, precision, recall per language (FR/EN/DE)
and per intent type (control/chat).
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
except ImportError as e:
    print(f"Error: Missing required dependencies.")
    print(f"Install with: pip install sentence-transformers scikit-learn")
    raise e


def load_dataset(path: Path) -> List[Dict]:
    """Load dataset from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def train_classifier(model_name: str, train_examples: List[Dict]):
    """Train a classifier on the training set."""
    print(f"Loading model: {model_name}...", end=" ", flush=True)
    model = SentenceTransformer(model_name)
    print("✓")

    print("Generating embeddings for training...", end=" ", flush=True)
    train_texts = [ex["text"] for ex in train_examples]
    train_labels = [ex["label"] for ex in train_examples]
    embeddings = model.encode(train_texts, show_progress_bar=False)
    print("✓")

    print("Training logistic regression...", end=" ", flush=True)
    y = np.array([1 if label == "control" else 0 for label in train_labels])
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(embeddings, y)
    print("✓")

    return model, classifier


def predict_examples(model, classifier, examples: List[Dict]) -> Tuple[List[str], List[float]]:
    """Predict on examples."""
    texts = [ex["text"] for ex in examples]
    embeddings = model.encode(texts, show_progress_bar=False)
    probs = classifier.predict_proba(embeddings)
    confidences = probs[:, 1]
    predictions = ["control" if c >= 0.5 else "chat" for c in confidences]
    return predictions, confidences


def compute_metrics(y_true, y_pred, label=""):
    """Compute precision, recall, F1, FP/FN rates."""
    y_true_bin = np.array([1 if label == "control" else 0 for label in y_true])
    y_pred_bin = np.array([1 if pred == "control" else 0 for pred in y_pred])

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )

    accuracy = (y_true_bin == y_pred_bin).mean()

    # Confusion matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        fp_rate = 0.0
        fn_rate = 0.0

    return {
        "label": label,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "total": len(y_true),
    }


def analyze_by_language(examples: List[Dict], predictions: List[str]) -> Dict[str, Dict]:
    """Analyze results broken down by language."""
    results_by_lang = {}

    for lang in ["fr", "en", "de"]:
        lang_examples = [ex for ex in examples if ex.get("lang") == lang]
        if not lang_examples:
            continue

        lang_indices = [i for i, ex in enumerate(examples) if ex.get("lang") == lang]
        lang_predictions = [predictions[i] for i in lang_indices]
        lang_true = [ex["label"] for ex in lang_examples]

        results_by_lang[lang] = compute_metrics(lang_true, lang_predictions, label=lang.upper())

    return results_by_lang


def analyze_by_intent(examples: List[Dict], predictions: List[str]) -> Dict[str, Dict]:
    """Analyze results broken down by intent type."""
    results_by_intent = {}

    for intent in ["control", "chat"]:
        intent_examples = [ex for ex in examples if ex["label"] == intent]
        if not intent_examples:
            continue

        intent_indices = [i for i, ex in enumerate(examples) if ex["label"] == intent]
        intent_predictions = [predictions[i] for i in intent_indices]
        intent_true = [ex["label"] for ex in intent_examples]

        results_by_intent[intent] = compute_metrics(intent_true, intent_predictions, label=intent.capitalize())

    return results_by_intent


def analyze_by_lang_and_intent(examples: List[Dict], predictions: List[str]) -> Dict[str, Dict[str, Dict]]:
    """Analyze results broken down by both language and intent."""
    results = defaultdict(lambda: defaultdict(dict))

    for lang in ["fr", "en", "de"]:
        for intent in ["control", "chat"]:
            filtered = [
                (ex, predictions[i])
                for i, ex in enumerate(examples)
                if ex.get("lang") == lang and ex["label"] == intent
            ]

            if not filtered:
                continue

            true_labels = [ex["label"] for ex, _ in filtered]
            preds = [pred for _, pred in filtered]

            results[lang][intent] = compute_metrics(true_labels, preds, label=f"{lang.upper()}-{intent}")

    return results


def print_overall_stats(examples: List[Dict], predictions: List[str]):
    """Print overall statistics."""
    y_true = [ex["label"] for ex in examples]
    metrics = compute_metrics(y_true, predictions, label="Overall")

    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)
    print(f"Total Examples:     {metrics['total']}")
    print(f"Accuracy:           {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Precision:          {metrics['precision']:.3f}")
    print(f"Recall:             {metrics['recall']:.3f}")
    print(f"F1 Score:           {metrics['f1']:.3f}")
    print(f"False Positive Rate: {metrics['fp_rate']:.3f} ({metrics['fp_rate']*100:.1f}%)")
    print(f"False Negative Rate: {metrics['fn_rate']:.3f} ({metrics['fn_rate']*100:.1f}%)")


def print_by_language_table(results_by_lang: Dict[str, Dict]):
    """Print per-language breakdown."""
    print("\n" + "=" * 100)
    print("PERFORMANCE BY LANGUAGE")
    print("=" * 100)

    header = f"{'Language':<12} {'Total':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FP Rate':>9} {'FN Rate':>9}"
    print(header)
    print("-" * 100)

    for lang in ["fr", "en", "de"]:
        if lang not in results_by_lang:
            continue
        r = results_by_lang[lang]
        row = (
            f"{lang.upper():<12} "
            f"{r['total']:>7d} "
            f"{r['accuracy']:>7.3f} "
            f"{r['precision']:>7.3f} "
            f"{r['recall']:>7.3f} "
            f"{r['f1']:>7.3f} "
            f"{r['fp_rate']:>9.3f} "
            f"{r['fn_rate']:>9.3f}"
        )
        print(row)

    print("=" * 100)


def print_by_intent_table(results_by_intent: Dict[str, Dict]):
    """Print per-intent breakdown."""
    print("\n" + "=" * 100)
    print("PERFORMANCE BY INTENT TYPE")
    print("=" * 100)

    header = f"{'Intent':<12} {'Total':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FP Rate':>9} {'FN Rate':>9}"
    print(header)
    print("-" * 100)

    for intent in ["control", "chat"]:
        if intent not in results_by_intent:
            continue
        r = results_by_intent[intent]
        row = (
            f"{intent.capitalize():<12} "
            f"{r['total']:>7d} "
            f"{r['accuracy']:>7.3f} "
            f"{r['precision']:>7.3f} "
            f"{r['recall']:>7.3f} "
            f"{r['f1']:>7.3f} "
            f"{r['fp_rate']:>9.3f} "
            f"{r['fn_rate']:>9.3f}"
        )
        print(row)

    print("=" * 100)


def print_detailed_breakdown(results: Dict[str, Dict[str, Dict]]):
    """Print detailed language × intent breakdown."""
    print("\n" + "=" * 110)
    print("DETAILED BREAKDOWN: LANGUAGE × INTENT")
    print("=" * 110)

    header = f"{'Language-Intent':<18} {'Total':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FP Rate':>9} {'FN Rate':>9}"
    print(header)
    print("-" * 110)

    for lang in ["fr", "en", "de"]:
        if lang not in results:
            continue
        for intent in ["control", "chat"]:
            if intent not in results[lang]:
                continue
            r = results[lang][intent]
            row = (
                f"{lang.upper()}-{intent:<12} "
                f"{r['total']:>7d} "
                f"{r['accuracy']:>7.3f} "
                f"{r['precision']:>7.3f} "
                f"{r['recall']:>7.3f} "
                f"{r['f1']:>7.3f} "
                f"{r['fp_rate']:>9.3f} "
                f"{r['fn_rate']:>9.3f}"
            )
            print(row)

    print("=" * 110)


def print_sample_errors(examples: List[Dict], predictions: List[str], max_samples: int = 10):
    """Print sample misclassifications per language."""
    print("\n" + "=" * 100)
    print(f"SAMPLE MISCLASSIFICATIONS (up to {max_samples} per language)")
    print("=" * 100)

    for lang in ["fr", "en", "de"]:
        errors = [
            (ex, predictions[i])
            for i, ex in enumerate(examples)
            if ex.get("lang") == lang and ex["label"] != predictions[i]
        ]

        if not errors:
            print(f"\n{lang.upper()}: No errors! ✓")
            continue

        print(f"\n{lang.upper()}: {len(errors)} errors")
        print("-" * 100)

        for ex, pred in errors[:max_samples]:
            true_label = ex["label"]
            print(f"  [{true_label:>7s} → {pred:<7s}] {ex['text']}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results by language and intent type"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model to use for analysis"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to JSONL dataset file"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)"
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Show sample misclassifications"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    examples = load_dataset(args.dataset)
    print(f"✓ Loaded {len(examples)} examples")

    # Show dataset composition
    lang_counts = {lang: sum(1 for ex in examples if ex.get("lang") == lang) for lang in ["fr", "en", "de"]}
    intent_counts = {intent: sum(1 for ex in examples if ex["label"] == intent) for intent in ["control", "chat"]}

    print(f"\nDataset Composition:")
    print(f"  Languages: FR={lang_counts['fr']}, EN={lang_counts['en']}, DE={lang_counts['de']}")
    print(f"  Intents:   Control={intent_counts['control']}, Chat={intent_counts['chat']}")

    # Split train/test
    train_examples, test_examples = train_test_split(
        examples, test_size=args.test_split, random_state=42,
        stratify=[e["label"] for e in examples]
    )
    print(f"\nSplit: Training={len(train_examples)}, Testing={len(test_examples)}")

    # Train classifier
    print(f"\nTraining classifier with {args.model}...")
    model, classifier = train_classifier(args.model, train_examples)

    # Predict on test set
    print("Evaluating on test set...", end=" ", flush=True)
    predictions, confidences = predict_examples(model, classifier, test_examples)
    print("✓")

    # Analyze results
    print_overall_stats(test_examples, predictions)

    results_by_lang = analyze_by_language(test_examples, predictions)
    print_by_language_table(results_by_lang)

    results_by_intent = analyze_by_intent(test_examples, predictions)
    print_by_intent_table(results_by_intent)

    results_detailed = analyze_by_lang_and_intent(test_examples, predictions)
    print_detailed_breakdown(results_detailed)

    if args.show_errors:
        print_sample_errors(test_examples, predictions)

    print("\n" + "=" * 100)
    print("✓ Analysis complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
