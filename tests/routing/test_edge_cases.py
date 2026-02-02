#!/usr/bin/env python3
"""
Test trained model on edge cases to find failures.
"""

import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_v2 import TrainedIntentClassifier, load_dataset
from sklearn.model_selection import train_test_split


def test_edge_cases():
    """Test model on edge cases."""
    # Load training data
    train_path = Path("data/routing/synthetic_examples.jsonl")
    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        return

    print("Loading training data...")
    train_examples = load_dataset(train_path)
    train_texts = [ex["text"] for ex in train_examples]
    train_labels = [ex["label"] for ex in train_examples]

    # Train classifier
    print("\nTraining classifier on all-MiniLM-L6-v2...")
    classifier = TrainedIntentClassifier("all-MiniLM-L6-v2")
    classifier.train(train_texts, train_labels)

    # Load edge cases
    edge_path = Path("data/routing/edge_cases.jsonl")
    if not edge_path.exists():
        print(f"Error: Edge cases not found at {edge_path}")
        return

    print("\nLoading edge cases...")
    edge_cases = load_dataset(edge_path)

    # Test each case
    print("\n" + "=" * 100)
    print("Edge Case Analysis")
    print("=" * 100)
    print(f"{'Text':<50} {'True':<10} {'Pred':<10} {'Conf':<8} {'Result'}")
    print("-" * 100)

    errors = []
    for example in edge_cases:
        text = example["text"]
        true_label = example["label"]

        # Predict
        pred_label, confidence, latency = classifier.predict_single(text)

        # Check if correct
        correct = "✓" if pred_label == true_label else "✗"
        if pred_label != true_label:
            errors.append({
                "text": text,
                "true": true_label,
                "pred": pred_label,
                "confidence": confidence
            })

        print(f"{text:<50} {true_label:<10} {pred_label:<10} {confidence:>7.3f} {correct}")

    print("=" * 100)

    # Summary
    accuracy = 1.0 - (len(errors) / len(edge_cases))
    print(f"\nEdge Case Accuracy: {accuracy:.1%} ({len(edge_cases) - len(errors)}/{len(edge_cases)})")

    if errors:
        print(f"\n⚠️  {len(errors)} Error(s) Found:")
        for err in errors:
            print(f"  - \"{err['text']}\" → predicted {err['pred']} (should be {err['true']}, conf={err['confidence']:.3f})")
    else:
        print("\n✓ All edge cases passed!")
        print("\n⚠️  Note: 100% on edge cases might mean they're still too easy.")
        print("   Real production data will likely have more ambiguity.")


if __name__ == "__main__":
    test_edge_cases()
