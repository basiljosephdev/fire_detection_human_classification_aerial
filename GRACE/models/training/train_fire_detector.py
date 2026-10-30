"""
train_fire_detector.py — Trains the fire/smoke detection model.

RUN THIS ON A PC OR CLOUD GPU, NOT THE JETSON NANO. The Nano is an inference
device; training YOLOv8 needs far more compute/RAM than it has, and its
stock JetPack ships old Python/CUDA versions that don't support current
`ultralytics`/PyTorch anyway.

If you don't have a local NVIDIA GPU, this will still run on CPU but very
slowly (100 epochs over ~21,500 images could take many hours to days) —
the script prints which device it's actually using at startup so you're not
guessing.

Dataset: DFire (fire + smoke, already in YOLO bbox format). The Kaggle
re-upload "Smoke-Fire-Detection-YOLO" is the same dataset. Some re-uploaded
copies ship a data.yaml whose `path:` field points at the ORIGINAL
uploader's own local folder (e.g. a stray "/kaggle/working/..." path) —
this won't exist on your machine either. Use --data_root to override it
rather than hand-editing the yaml:

Usage:
    pip install ultralytics
    python3 train_fire_detector.py --data /path/to/dfire/data.yaml --epochs 100

    # If the yaml's `path:` field is wrong/foreign, point at the real
    # extracted dataset folder instead (the one actually containing your
    # train/val/test subfolders) and this script will patch a corrected
    # copy of the yaml before training:
    python3 train_fire_detector.py --data /path/to/dfire/data.yaml \\
        --data_root "D:\\Downloads\\Compressed\\DFire" --epochs 100
"""

import argparse
import os

import torch
import yaml
from ultralytics import YOLO


def _resolve_data_yaml(data_path, data_root):
    """If --data_root is given, rewrite the yaml's `path:` field to point
    there and return a path to the corrected copy instead of the original."""
    if not data_root:
        return data_path

    with open(data_path) as f:
        cfg = yaml.safe_load(f)

    print(f"[train_fire_detector] Overriding data.yaml path: '{cfg.get('path')}' -> '{data_root}'")
    cfg["path"] = data_root

    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if rel:
            resolved = os.path.normpath(os.path.join(data_root, rel))
            exists = os.path.isdir(resolved)
            print(f"  {split}: {resolved}  exists={exists}")
            if not exists:
                print(f"  [warning] '{split}' path doesn't exist — check --data_root and the yaml's "
                      f"'{split}:' value are actually compatible before trusting this run.")

    fixed_path = os.path.join(os.path.dirname(os.path.abspath(data_path)), "_data_fixed.yaml")
    with open(fixed_path, "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"[train_fire_detector] Wrote corrected yaml to {fixed_path}\n")
    return fixed_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to DFire data.yaml")
    parser.add_argument("--data_root", default=None,
                         help="Override the yaml's 'path:' field with this folder if it's wrong/foreign.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    # yolov8n (nano) is the right size class for a Jetson Nano at the receiving
    # end — don't reach for yolov8m/l/x, they won't run at usable FPS on-device.
    parser.add_argument("--base_model", default="yolov8n.pt")
    parser.add_argument("--device", default=None,
                         help="Force a device, e.g. 'cpu' or '0'. Default: auto (GPU if available, else CPU).")
    args = parser.parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"[train_fire_detector] Using device: {device}"
          + ("" if device != "cpu" else "  (NO GPU DETECTED — this will be slow. "
                                          "Check your CUDA/PyTorch install if you expected a GPU here.)"))

    data_path = _resolve_data_yaml(args.data, args.data_root)

    model = YOLO(args.base_model)
    model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project="runs/fire_detector",
        name="train",
    )

    metrics = model.val()
    print(f"[train_fire_detector] Validation mAP50: {metrics.box.map50:.3f}")
    print("[train_fire_detector] Best weights saved under runs/fire_detector/train/weights/best.pt")
    print("[train_fire_detector] Next step: python3 export_onnx.py --weights runs/fire_detector/train/weights/best.pt --out fire_detector.onnx")


if __name__ == "__main__":
    main()
