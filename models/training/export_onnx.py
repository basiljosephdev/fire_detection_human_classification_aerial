"""
export_onnx.py — Exports a trained YOLOv8 .pt checkpoint to ONNX.

RUN THIS ON THE SAME PC/GPU YOU TRAINED ON. ONNX is portable, so this file
is what you copy over to the Jetson Nano afterwards — the .engine file
built FROM this ONNX must be built ON the Nano itself (see
build_tensorrt_engine.sh), because TensorRT engines are locked to the exact
GPU/TensorRT version they were built on.

opset=12 is chosen deliberately: Jetson Nano ships an old JetPack (4.6.x)
with TensorRT ~8.2, which is fussy about very new opsets. If `trtexec` on
the Nano complains about unsupported ops later, dropping to opset=11 is
the next thing to try.

Usage:
    python3 export_onnx.py --weights runs/fire_detector/train/weights/best.pt --out fire_detector.onnx
    python3 export_onnx.py --weights runs/custom_posture_detector/train/weights/best.pt --out custom_posture_detector.onnx
"""

import argparse
import shutil

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument("--out", required=True, help="Output .onnx filename")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,   # runs onnx-simplifier, helps trtexec parse it cleanly
        dynamic=False,   # fixed input shape — simpler and faster for a single Nano camera feed
    )

    shutil.move(str(exported_path), args.out)
    print(f"[export_onnx] Wrote {args.out}")
    print("[export_onnx] Copy this file to the Jetson Nano, then run build_tensorrt_engine.sh ON THE NANO.")


if __name__ == "__main__":
    main()
