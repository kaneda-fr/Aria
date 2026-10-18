#!/usr/bin/env python3
"""
Quick test script for the self-learning router.
"""

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from app.routing import get_router

# Test examples
test_cases = [
    # CHAT examples (should be classified as chat)
    ("bonjour", "chat"),
    ("hello", "chat"),
    ("merci", "chat"),
    ("thank you", "chat"),
    ("quelle heure est-il?", "chat"),
    ("what time is it?", "chat"),
    ("comment ça va?", "chat"),
    ("how are you?", "chat"),

    # CONTROL examples (should be classified as control)
    ("allume la lumière", "control"),
    ("turn on the light", "control"),
    ("éteins la lampe", "control"),
    ("turn off the lamp", "control"),
    ("mets la lumière à 50%", "control"),
    ("set light to 50%", "control"),
    ("ouvre les volets", "control"),
    ("open the shutters", "control"),

    # Edge cases (unknown or borderline)
    ("lumière", "unknown/control"),  # Just the word, ambiguous
    ("salon", "unknown/control"),     # Just room name
]

def main():
    print("=" * 80)
    print("ARIA Self-Learning Router Test")
    print("=" * 80)
    print()

    # Initialize router
    print("Initializing router...")
    router = get_router(
        control_threshold=0.70,
        chat_threshold=0.70,
        unknown_threshold=0.50,
        use_seed_data=True
    )

    stats = router.get_stats()
    print(f"Router initialized:")
    print(f"  Model: {stats['model_name']}")
    print(f"  Centroid distance: {stats['centroid_distance']:.3f}")
    print(f"  Centroids initialized: {stats['centroids_initialized']}")
    print()

    # Run tests
    print("Running classification tests...")
    print("=" * 80)

    correct = 0
    total = 0

    for text, expected in test_cases:
        decision = router.classify(text)
        total += 1

        # Check if classification matches expected
        is_correct = (
            decision.mode == expected or
            expected.startswith(decision.mode) or  # "unknown/control" allows either
            decision.mode in expected.split("/")
        )

        if is_correct:
            correct += 1
            result = "✓"
        else:
            result = "✗"

        print(f"{result} '{text[:40]:<40}' → {decision.mode.upper():<8} "
              f"(conf={decision.confidence:.2f}, "
              f"d_ctrl={decision.distance_to_control:.3f}, "
              f"d_chat={decision.distance_to_chat:.3f})")

    print("=" * 80)
    print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print()

    if correct == total:
        print("✓ All tests passed!")
    elif correct / total >= 0.8:
        print("⚠ Most tests passed, but some edge cases failed.")
    else:
        print("✗ Many tests failed - router may need tuning.")

if __name__ == "__main__":
    main()
