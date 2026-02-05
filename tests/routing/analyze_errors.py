#!/usr/bin/env python3
"""
Analyze which examples the model gets wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark_v2 import TrainedIntentClassifier, load_dataset
from sklearn.model_selection import train_test_split
import numpy as np


def analyze_errors():
    """Find and display misclassified examples."""
    # Load dataset
    dataset_path = Path("data/routing/synthetic_with_edges.jsonl")
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("Loading dataset...")
    examples = load_dataset(dataset_path)

    # Split same way as benchmark
    train_examples, test_examples = train_test_split(
        examples, test_size=0.2, random_state=42, stratify=[e["label"] for e in examples]
    )

    # Train both models
    models = ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2"]

    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Error Analysis: {model_name}")
        print(f"{'='*80}")

        # Train classifier
        classifier = TrainedIntentClassifier(model_name)
        train_texts = [ex["text"] for ex in train_examples]
        train_labels = [ex["label"] for ex in train_examples]
        classifier.train(train_texts, train_labels)

        # Predict on test set
        test_texts = [ex["text"] for ex in test_examples]
        test_labels = [ex["label"] for ex in test_examples]

        print("Predicting on test set...")
        predictions = []
        confidences = []
        for text in test_texts:
            pred, conf, _ = classifier.predict_single(text)
            predictions.append(pred)
            confidences.append(conf)

        # Find errors
        errors = []
        for i, (text, true_label, pred_label, conf) in enumerate(
            zip(test_texts, test_labels, predictions, confidences)
        ):
            if true_label != pred_label:
                errors.append({
                    "text": text,
                    "true": true_label,
                    "pred": pred_label,
                    "confidence": conf,
                    "lang": test_examples[i]["lang"]
                })

        # Display errors
        print(f"\n✗ Found {len(errors)} errors ({len(errors)/len(test_examples)*100:.1f}%)\n")

        # Group by error type
        false_positives = [e for e in errors if e["true"] == "chat" and e["pred"] == "control"]
        false_negatives = [e for e in errors if e["true"] == "control" and e["pred"] == "chat"]

        if false_positives:
            print(f"False Positives (chat → control): {len(false_positives)} ⚠️ CRITICAL")
            print("-" * 80)
            for err in false_positives:
                print(f"  [{err['lang']}] \"{err['text']}\"")
                print(f"       → Predicted: control (conf={err['confidence']:.3f}, should be chat)")
            print()

        if false_negatives:
            print(f"False Negatives (control → chat): {len(false_negatives)}")
            print("-" * 80)
            for err in false_negatives:
                print(f"  [{err['lang']}] \"{err['text']}\"")
                print(f"       → Predicted: chat (conf={err['confidence']:.3f}, should be control)")
            print()

        # Statistics
        if errors:
            avg_conf = np.mean([e["confidence"] for e in errors])
            print(f"Average confidence on errors: {avg_conf:.3f}")
            print(f"  (Lower confidence = model was uncertain)")


if __name__ == "__main__":
    analyze_errors()
