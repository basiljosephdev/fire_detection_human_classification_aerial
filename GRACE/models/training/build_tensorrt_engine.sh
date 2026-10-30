#!/usr/bin/env bash
# build_tensorrt_engine.sh — RUN THIS ON THE JETSON NANO ITSELF.
#
# Converts an ONNX model (built by export_onnx.py on your PC and copied
# over, e.g. via scp) into a .engine file optimized for this specific
# Nano's GPU and the TensorRT version installed by its JetPack image.
# This step CANNOT be done on a PC and copied over — the .engine format is
# tied to the exact TensorRT build/GPU it was compiled on.
#
# `trtexec` ships with JetPack by default at /usr/src/tensorrt/bin/trtexec.
# If it's not on your PATH, use the full path instead.
#
# --fp16 uses half-precision math, which the Nano's Maxwell GPU supports
# and which roughly doubles throughput over FP32 with minimal accuracy
# loss for detection models — this is the setting the README's "28-32 FPS
# with an accelerator" territory assumes you're in.
#
# --workspace is capped low (512 MB) because the Nano only has 4GB of RAM
# *shared* between CPU and GPU — a large workspace can starve everything
# else running on the board (camera capture, mavlink, etc). Raise it only
# if the build fails complaining it needs more space, and only in small
# steps.
#
# Usage:
#   ./build_tensorrt_engine.sh fire_detector.onnx fire_detector.engine

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input.onnx> <output.engine>"
    exit 1
fi

ONNX_PATH="$1"
ENGINE_PATH="$2"
TRTEXEC="$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)"

echo "Building TensorRT engine on this Jetson Nano..."
echo "  ONNX in:    $ONNX_PATH"
echo "  Engine out: $ENGINE_PATH"

"$TRTEXEC" \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --fp16 \
    --workspace=512 \
    --verbose

echo "Done. Move $ENGINE_PATH into edge/models/ and point settings.yaml at it."
