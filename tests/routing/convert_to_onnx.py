#!/usr/bin/env python3
"""
Convert sentence-transformers model to quantized ONNX format.

This converts the model to ONNX and applies INT8 quantization for faster CPU inference.
"""

import argparse
from pathlib import Path
import shutil


def convert_to_onnx(model_name: str, output_dir: Path, quantize: bool = True):
    """
    Convert sentence-transformers model to ONNX format.

    Args:
        model_name: HuggingFace model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        output_dir: Directory to save ONNX model
        quantize: Whether to apply INT8 quantization
    """
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: Missing dependencies.")
        print("Install with: pip install optimum[onnxruntime] transformers")
        return

    print(f"Converting {model_name} to ONNX...")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to ONNX
    print("\n[1/3] Exporting to ONNX format...")
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_name,
        export=True,
        provider="CPUExecutionProvider"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quantize:
        print("[2/3] Applying INT8 quantization...")
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            # Create quantizer
            quantizer = ORTQuantizer.from_pretrained(model)

            # Use AVX512-VNNI quantization config (optimal for Xeon)
            # Falls back to AVX2 if AVX512 not available
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

            # Quantize
            quantizer.quantize(save_dir=output_dir, quantization_config=qconfig)

            print("✓ INT8 quantization applied (AVX512-VNNI)")
        except Exception as e:
            print(f"⚠️  Quantization failed: {e}")
            print("Saving unquantized ONNX model...")
            model.save_pretrained(output_dir)
    else:
        print("[2/3] Skipping quantization...")
        model.save_pretrained(output_dir)

    # Save tokenizer
    print("[3/3] Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)

    print(f"\n✓ Model saved to: {output_dir}")

    # Show file sizes
    onnx_files = list(output_dir.glob("*.onnx"))
    if onnx_files:
        for f in onnx_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name}: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert sentence-transformers model to quantized ONNX"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model ID (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/all-MiniLM-L6-v2-onnx-int8"),
        help="Output directory for ONNX model"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization"
    )

    args = parser.parse_args()

    convert_to_onnx(args.model, args.output, quantize=not args.no_quantize)


if __name__ == "__main__":
    main()
