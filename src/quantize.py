"""
Model Optimization & Quantization Pipeline
Exports PyTorch YOLOv8 weights to ONNX format and applies INT8 dynamic quantization
for optimized, low-latency CPU inference.
"""

import argparse
import os
import sys
from typing import Dict, Any


def get_file_size_mb(path: str) -> float:
    """Returns file size in megabytes (MB)."""
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def export_to_onnx(
    pt_model_path: str,
    output_onnx_path: str,
    img_size: int = 640,
    opset: int = 17,
    simplify: bool = True
) -> str:
    """
    Exports YOLOv8 PyTorch (.pt) weights to ONNX format.
    """
    from ultralytics import YOLO

    print(f"[*] Loading PyTorch model from: {pt_model_path}")
    model = YOLO(pt_model_path)

    print(f"[*] Exporting to ONNX (imgsz={img_size}, opset={opset}, simplify={simplify})...")
    exported_file = model.export(
        format="onnx",
        imgsz=img_size,
        opset=opset,
        simplify=simplify,
        dynamic=False
    )

    if exported_file != output_onnx_path and os.path.exists(exported_file):
        os.makedirs(os.path.dirname(output_onnx_path) or ".", exist_ok=True)
        import shutil
        shutil.move(exported_file, output_onnx_path)

    print(f"[+] ONNX export succeeded: {output_onnx_path} ({get_file_size_mb(output_onnx_path):.2f} MB)")
    return output_onnx_path


def quantize_to_int8(
    input_onnx_path: str,
    output_int8_path: str
) -> str:
    """
    Applies INT8 Dynamic Quantization using ONNX Runtime.
    Quantizes weights to 8-bit integers while preserving operator graphs for CPU execution.
    """
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if not os.path.exists(input_onnx_path):
        raise FileNotFoundError(f"Input ONNX model not found: {input_onnx_path}")

    print(f"[*] Starting INT8 Dynamic Quantization on: {input_onnx_path}...")
    os.makedirs(os.path.dirname(output_int8_path) or ".", exist_ok=True)

    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=output_int8_path,
        weight_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=False
    )

    orig_size = get_file_size_mb(input_onnx_path)
    quant_size = get_file_size_mb(output_int8_path)
    compression = (1.0 - (quant_size / orig_size)) * 100.0 if orig_size > 0 else 0.0

    print(f"[+] INT8 Quantization completed successfully!")
    print(f"    - Original ONNX size:   {orig_size:.2f} MB")
    print(f"    - Quantized INT8 size: {quant_size:.2f} MB")
    print(f"    - Memory reduction:     {compression:.1f}%")

    # Sanity check: verify loadability with ONNX Runtime
    session = ort.InferenceSession(output_int8_path, providers=["CPUExecutionProvider"])
    print(f"[+] Model graph verified with {len(session.get_inputs())} inputs, {len(session.get_outputs())} outputs.")

    return output_int8_path


def optimize_pipeline(pt_path: str, output_dir: str = "outputs/models") -> Dict[str, str]:
    """Runs the full PyTorch -> ONNX -> Quantized INT8 pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pt_path))[0]

    onnx_path = os.path.join(output_dir, f"{base_name}.onnx")
    int8_path = os.path.join(output_dir, f"{base_name}_int8.onnx")

    export_to_onnx(pt_path, onnx_path)
    quantize_to_int8(onnx_path, int8_path)

    return {
        "pytorch": pt_path,
        "onnx": onnx_path,
        "onnx_int8": int8_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and quantize YOLOv8 models to ONNX and INT8")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch .pt weights")
    parser.add_argument("--output-dir", type=str, default="outputs/models", help="Directory for optimized models")
    args = parser.parse_args()

    optimize_pipeline(args.model, args.output_dir)
