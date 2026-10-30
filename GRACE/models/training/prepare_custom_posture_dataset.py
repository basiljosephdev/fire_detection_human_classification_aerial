"""
prepare_custom_posture_dataset.py — Converts Atomic dataset annotations
into a dedicated 3-class YOLO posture dataset:
  0: Lying
  1: Sitting
  2: Standing

Human-Human interaction classes (handshaking, hugging, etc.) are strictly excluded
so that the YOLO model learns clean posture representations without multi-person
interaction confusion.
"""

import os
import glob
import random
import shutil
from collections import Counter
from PIL import Image

def get_image_size(img_path):
    """
    Fast dimension extraction via PIL without full pixel decompression.
    Falls back gracefully if needed.
    """
    try:
        with Image.open(img_path) as im:
            return im.size  # (width, height)
    except Exception:
        pass
    try:
        import cv2  # type: ignore
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except Exception:
        pass
    return None, None

TARGET_CLASSES = ["Lying", "Sitting", "Standing"]

# Clean posture mapping: only posture states, strictly no human interactions
CLASS_MAPPING = {
    # 0: Lying
    "lying": 0,
    # 1: Sitting
    "sitting": 1,
    # 2: Standing (includes upright movement postures for full frame coverage)
    "standing": 2,
    "walking": 2,
    "running": 2,
    "waving": 2,
}

# Explicitly excluded interaction labels to prevent noise
INTERACTION_CLASSES = {
    "handshaking",
    "hugging",
    "interaction",
}


def parse_and_convert_annotation(anno_path, img_w, img_h, min_box_dim=6):
    """
    Parses an annotation file and converts valid posture bounding boxes to YOLO format.
    Excludes any human interaction classes or unmapped labels.
    """
    yolo_lines = []
    class_counts = Counter()

    with open(anno_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue

            raw_label = parts[1].strip().lower()

            # Strictly skip human interaction classes and unmapped labels
            if raw_label in INTERACTION_CLASSES or raw_label not in CLASS_MAPPING:
                continue

            class_id = CLASS_MAPPING[raw_label]

            try:
                xmin = float(parts[3].strip())
                ymin = float(parts[4].strip())
                box_w = float(parts[5].strip())
                box_h = float(parts[6].strip())
            except ValueError:
                continue

            # Skip invalid, inverted or degenerate boxes
            if box_w < min_box_dim or box_h < min_box_dim:
                continue

            # Clip box to image boundaries
            x1 = max(0.0, min(float(img_w), xmin))
            y1 = max(0.0, min(float(img_h), ymin))
            x2 = max(0.0, min(float(img_w), xmin + box_w))
            y2 = max(0.0, min(float(img_h), ymin + box_h))

            w_clipped = x2 - x1
            h_clipped = y2 - y1
            if w_clipped < min_box_dim or h_clipped < min_box_dim:
                continue

            # Convert to normalized YOLO format (center_x, center_y, width, height)
            center_x = (x1 + w_clipped / 2.0) / img_w
            center_y = (y1 + h_clipped / 2.0) / img_h
            norm_w = w_clipped / img_w
            norm_h = h_clipped / img_h

            center_x = max(0.0, min(1.0, center_x))
            center_y = max(0.0, min(1.0, center_y))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
            class_counts[class_id] += 1

    return yolo_lines, class_counts


def build_dataset(dataset_root, output_dir, val_split=0.2, seed=42, include_human_human=False):
    """
    Builds the clean 3-class YOLO posture dataset.
    By default, uses 'Atomic' individual postures and excludes 'Human_Human' interaction subsets.
    """
    random.seed(seed)
    
    subsets = ["Atomic"]
    if include_human_human:
        subsets.append("Human_Human")

    samples = []
    for sub in subsets:
        anno_dir = os.path.join(dataset_root, sub, "annotations_final")
        img_dir = os.path.join(dataset_root, sub, "extracted_frames")

        if not os.path.exists(anno_dir) or not os.path.exists(img_dir):
            print(f"[warning] Subset directory missing: {sub}")
            continue

        anno_files = glob.glob(os.path.join(anno_dir, "*.txt"))
        for anno_path in anno_files:
            stem = os.path.splitext(os.path.basename(anno_path))[0]
            img_path = os.path.join(img_dir, f"{stem}.jpg")
            if os.path.exists(img_path):
                samples.append((img_path, anno_path, f"{sub}_{stem}"))

    print(f"[prepare_custom_posture_dataset] Found {len(samples)} valid image-annotation pairs across subsets: {subsets}")
    if not samples:
        raise RuntimeError("No matching image-annotation pairs found!")

    random.shuffle(samples)

    val_count = int(len(samples) * val_split)
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    # Clean and recreate directories to prevent stale or cross-split leftover files
    for split in ["train", "val"]:
        img_split_dir = os.path.join(output_dir, "images", split)
        lbl_split_dir = os.path.join(output_dir, "labels", split)
        if os.path.exists(img_split_dir):
            shutil.rmtree(img_split_dir, ignore_errors=True)
        if os.path.exists(lbl_split_dir):
            shutil.rmtree(lbl_split_dir, ignore_errors=True)
        os.makedirs(img_split_dir, exist_ok=True)
        os.makedirs(lbl_split_dir, exist_ok=True)

    split_stats = {"train": Counter(), "val": Counter()}

    def process_split(sample_list, split_name):
        converted_count = 0
        for img_path, anno_path, unique_stem in sample_list:
            img_w, img_h = get_image_size(img_path)
            if img_w is None or img_h is None:
                continue

            yolo_lines, counts = parse_and_convert_annotation(anno_path, img_w, img_h)
            if not yolo_lines:
                continue

            dst_img = os.path.join(output_dir, "images", split_name, f"{unique_stem}.jpg")
            dst_label = os.path.join(output_dir, "labels", split_name, f"{unique_stem}.txt")

            shutil.copy2(img_path, dst_img)
            with open(dst_label, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")

            split_stats[split_name].update(counts)
            converted_count += 1
        return converted_count

    print("[prepare_custom_posture_dataset] Processing training split...")
    n_train = process_split(train_samples, "train")
    print("[prepare_custom_posture_dataset] Processing validation split...")
    n_val = process_split(val_samples, "val")

    # Create data.yaml
    abs_out_dir = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = f"""path: {abs_out_dir}
train: images/train
val: images/val

names:
  0: Lying
  1: Sitting
  2: Standing
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\n" + "=" * 60)
    print(f"Dataset successfully generated at: {output_dir}")
    print(f"  Train images: {n_train} | Val images: {n_val}")
    print("  Class Instance Distribution:")
    for cls_id, cls_name in enumerate(TARGET_CLASSES):
        train_c = split_stats["train"][cls_id]
        val_c = split_stats["val"][cls_id]
        total = train_c + val_c
        print(f"    [{cls_id}] {cls_name:10s} -> Total: {total:5d} (Train: {train_c:5d}, Val: {val_c:5d})")
    print(f"  data.yaml created at: {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="Dataset", help="Path to input Dataset folder")
    parser.add_argument("--output_dir", default="models/training/custom_posture_dataset", help="Output YOLO dataset path")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--include_human_human", action="store_true", help="Include Human_Human interaction subset")
    args = parser.parse_args()

    build_dataset(args.dataset_root, args.output_dir, args.val_split, include_human_human=args.include_human_human)
