"""
YOLOv8 Fine-Tuning Script for Helmet & No-Helmet Safety Detection
Trains YOLOv8n on public PPE/helmet dataset, logs validation metrics,
and saves the optimal weights to outputs/models/best.pt.
"""

import argparse
import os
import shutil
import sys
from ultralytics import YOLO


def train_model(
    data_yaml: str = "configs/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "cpu",
    output_dir: str = "outputs/models"
):
    print("============================================================")
    print(" HELMET SAFETY DETECTOR: YOLOV8 FINE-TUNING")
    print("============================================================")
    print(f"[*] Base Weights: {base_model}")
    print(f"[*] Dataset Config: {data_yaml}")
    print(f"[*] Target Epochs: {epochs}")
    print(f"[*] Batch Size: {batch_size}")
    print(f"[*] Input Resolution: {img_size}x{img_size}")
    print(f"[*] Device: {device}")

    # Load base model
    model = YOLO(base_model)

    # Train
    print("[*] Launching training pipeline...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        plots=True,
        save=True,
        save_period=10,
        name="helmet_v8_train",
        project="runs"
    )

    best_pt = os.path.join(results.save_dir, "weights", "best.pt")
    target_pt = os.path.join(output_dir, "best.pt")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(best_pt):
        shutil.copy(best_pt, target_pt)
        print(f"[+] Best weights saved to: {target_pt}")
    else:
        print(f"[!] Warning: Best weights not found at {best_pt}")

    # Validate on held-out test split
    print("\n[*] Validating on held-out test split...")
    test_metrics = model.val(data=data_yaml, split="test")
    print(f"    - Overall mAP@50:    {test_metrics.box.map50:.4f}")
    print(f"    - Overall mAP@50-95: {test_metrics.box.map:.4f}")

    return target_pt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on Helmet Safety Dataset")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or 0)")
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device
    )
