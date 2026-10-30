"""
train_custom_posture_detector.py — Trains the 3-class victim posture model
(Lying / Sitting / Standing) on the custom dataset.

Dataset classes:
  0: Lying
  1: Sitting
  2: Standing
"""

import argparse
import os

import torch
import yaml
from ultralytics import YOLO


def _resolve_data_yaml(data_path, data_root):
    if not data_root:
        return data_path

    with open(data_path) as f:
        cfg = yaml.safe_load(f)

    print(f"[train_custom_posture_detector] Overriding data.yaml path: '{cfg.get('path')}' -> '{data_root}'")
    cfg["path"] = data_root

    for split in ("train", "val"):
        rel = cfg.get(split)
        if rel:
            resolved = os.path.normpath(os.path.join(data_root, rel))
            exists = os.path.isdir(resolved)
            print(f"  {split}: {resolved}  exists={exists}")

    fixed_path = os.path.join(os.path.dirname(os.path.abspath(data_path)), "_data_fixed.yaml")
    with open(fixed_path, "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"[train_custom_posture_detector] Wrote corrected yaml to {fixed_path}\n")
    return fixed_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="models/training/custom_posture_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--base_model", default="models/training/yolov8n.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"[train_custom_posture_detector] Using device: {device}")

    data_path = _resolve_data_yaml(args.data, args.data_root)

    model = YOLO(args.base_model)
    model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project="runs/custom_posture_detector",
        name="train",
        degrees=15.0,
        fliplr=0.5,
    )

    metrics = model.val()
    print(f"[train_custom_posture_detector] Validation mAP50: {metrics.box.map50:.3f}")
    print("[train_custom_posture_detector] Per-class AP:", metrics.box.maps)
    print("[train_custom_posture_detector] Best weights saved under runs/custom_posture_detector/train/weights/best.pt")
    print("[train_custom_posture_detector] Next step: python export_onnx.py --weights runs/custom_posture_detector/train/weights/best.pt --out custom_posture_detector.onnx")


if __name__ == "__main__":
    main()
