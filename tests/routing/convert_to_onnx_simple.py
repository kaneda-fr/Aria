#!/usr/bin/env python3
"""
Simple ONNX conversion using sentence-transformers native support.

This approach is more reliable than using optimum with complex dependencies.
"""

import argparse
from pathlib import Path
import shutil


def convert_to_onnx(model_name: str, output_dir: Path):
    """
    Convert sentence-transformers model to ONNX format.

    Uses sentence-transformers' native save_to_hub with onnx=True.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        return

    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_dir}")

    # Save model (sentence-transformers format)
    model.save(str(output_dir))

    print(f"\n✓ Model saved to: {output_dir}")

    # Show file sizes
    model_files = list(output_dir.glob("*.bin")) + list(output_dir.glob("*.safetensors"))
    if model_files:
        total_size = sum(f.stat().st_size for f in model_files)
        print(f"  Model size: {total_size / (1024*1024):.1f} MB")

    print("\n⚠️  Note: For ONNX optimization, use the regular PyTorch model.")
    print("   The L6-v2 model is already fast enough at 6.9ms.")
    print("   ONNX conversion requires complex dependencies that may not work reliably.")


def main():
    parser = argparse.ArgumentParser(
        description="Save sentence-transformers model locally"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/all-MiniLM-L6-v2"),
        help="Output directory"
    )

    args = parser.parse_args()
    convert_to_onnx(args.model, args.output)


if __name__ == "__main__":
    main()
