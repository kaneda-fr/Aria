#!/usr/bin/env python3
"""
Manual ONNX export and quantization without optimum dependencies.

This uses torch.onnx for export and onnxruntime for quantization.
More reliable than optimum for servers with dependency issues.
"""

import argparse
import torch
from pathlib import Path
import json
import shutil


def export_to_onnx(model_name: str, output_dir: Path):
    """Export sentence-transformers model to ONNX manually."""
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("Error: Missing sentence-transformers or transformers")
        print("Install with: pip install sentence-transformers transformers")
        return False

    print(f"[1/4] Loading model: {model_name}")

    # Load the transformer model directly (not sentence-transformers wrapper)
    base_model_name = model_name.replace("sentence-transformers/", "")
    model = AutoModel.from_pretrained(f"sentence-transformers/{base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{base_model_name}")

    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[2/4] Exporting to ONNX...")

    # Create dummy input
    dummy_text = "This is a sample sentence."
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"

    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        str(onnx_path),
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'last_hidden_state': {0: 'batch', 1: 'sequence'}
        },
        opset_version=14
    )

    print(f"✓ ONNX model exported to: {onnx_path}")

    # Save tokenizer
    print("[3/4] Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    # Save config
    config = {
        "model_name": model_name,
        "max_seq_length": 256,
        "do_lower_case": False
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return True


def quantize_onnx(onnx_path: Path, output_path: Path):
    """Quantize ONNX model to INT8."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("⚠️  onnxruntime not installed, skipping quantization")
        print("   Model will work but won't be quantized")
        return False

    print("[4/4] Applying INT8 quantization...")

    try:
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            optimize_model=True
        )
        print(f"✓ Quantized model saved to: {output_path}")

        # Remove unquantized version
        onnx_path.unlink()

        return True
    except Exception as e:
        print(f"⚠️  Quantization failed: {e}")
        print(f"   Using unquantized ONNX model")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Manual ONNX export and quantization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model to convert"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/all-MiniLM-L6-v2-onnx-manual"),
        help="Output directory"
    )

    args = parser.parse_args()

    print(f"Converting {args.model} to quantized ONNX")
    print(f"Output: {args.output}\n")

    # Export to ONNX
    if not export_to_onnx(args.model, args.output):
        return

    # Quantize
    onnx_path = args.output / "model.onnx"
    quantized_path = args.output / "model_quantized.onnx"

    if onnx_path.exists():
        quantize_onnx(onnx_path, quantized_path)

    # Show file sizes
    print("\nModel files:")
    for f in args.output.glob("*.onnx"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

    print(f"\n✓ Conversion complete!")
    print(f"\nTo benchmark, use:")
    print(f"  python3 tests/routing/benchmark_onnx.py \\")
    print(f"    --onnx-model {args.output} \\")
    print(f"    --dataset data/routing/synthetic_with_edges.jsonl")


if __name__ == "__main__":
    main()
