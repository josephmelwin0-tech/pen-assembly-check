"""
Prepares the yolo_tip_dataset/ directory by cropping tip regions
from reference_images/ and test_images/ for all 4 tip classes:
  - empty_tip   (from state_0 and state_1)
  - refill_tip  (from state_2)
  - conecap_tip (from state_3)
  - topcap_tip  (from state_4)

Prevents train/val data leakage by respecting vid_train_ and vid_val_ prefixes.

Usage:
    python prepare_tip_dataset.py
"""

import os
import shutil
import cv2
from component_detector import ComponentDetector

REFERENCE_DIR = "reference_images"
TEST_DIR = "test_images"
OUTPUT_DIR = "yolo_tip_dataset"
VAL_SPLIT = 0.2

STATE_TO_TIP = {
    "state_0_barrel_only": "empty_tip",
    "state_1_backcap_on": "empty_tip",
    "state_2_refill_inserted": "refill_tip",
    "state_3_conecap_on": "conecap_tip",
    "state_4_topcap_on": "topcap_tip",
}

TIP_CLASSES = ["empty_tip", "refill_tip", "conecap_tip", "topcap_tip"]


def clear_and_make(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def process_and_save_tips(files, src_dir, dst_dir, detector, target_size=(192, 192)):
    os.makedirs(dst_dir, exist_ok=True)
    saved = 0
    for fname in files:
        src_path = os.path.join(src_dir, fname)
        img = cv2.imread(src_path)
        if img is None:
            continue

        tip_crop, has_bc, box = detector.extract_tip_and_endpoints(img)
        if tip_crop is not None and tip_crop.shape[0] >= 30 and tip_crop.shape[1] >= 30:
            resized = cv2.resize(tip_crop, target_size, interpolation=cv2.INTER_AREA)
            dst_path = os.path.join(dst_dir, fname)
            cv2.imwrite(dst_path, resized)
            saved += 1
    return saved


def main():
    if not os.path.isdir(REFERENCE_DIR):
        print(f"ERROR: '{REFERENCE_DIR}' not found.")
        return

    detector = ComponentDetector()
    clear_and_make(OUTPUT_DIR)

    # Initialize folders
    for split in ["train", "val", "test"]:
        for tclass in TIP_CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, tclass), exist_ok=True)

    summary = {c: {"train": 0, "val": 0, "test": 0} for c in TIP_CLASSES}

    # Process reference images (Train & Val)
    for state_name, tip_class in STATE_TO_TIP.items():
        src = os.path.join(REFERENCE_DIR, state_name)
        if not os.path.isdir(src):
            continue

        images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        vid_train = [f for f in images if f.startswith("vid_train_")]
        vid_val = [f for f in images if f.startswith("vid_val_")]
        static_photos = [f for f in images if not f.startswith("vid_")]

        # Split static photos if any
        n_val = int(len(static_photos) * VAL_SPLIT) if static_photos else 0
        static_val = static_photos[:n_val]
        static_train = static_photos[n_val:]

        train_files = vid_train + static_train
        val_files = vid_val + static_val

        tr_saved = process_and_save_tips(
            train_files, src, os.path.join(OUTPUT_DIR, "train", tip_class), detector
        )
        va_saved = process_and_save_tips(
            val_files, src, os.path.join(OUTPUT_DIR, "val", tip_class), detector
        )

        summary[tip_class]["train"] += tr_saved
        summary[tip_class]["val"] += va_saved

    # Process test images
    for state_name, tip_class in STATE_TO_TIP.items():
        test_src = os.path.join(TEST_DIR, state_name)
        if os.path.isdir(test_src):
            test_files = [f for f in os.listdir(test_src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            te_saved = process_and_save_tips(
                test_files, test_src, os.path.join(OUTPUT_DIR, "test", tip_class), detector
            )
            summary[tip_class]["test"] += te_saved

    print("=" * 60)
    print(" TIP DATASET PREPARATION SUMMARY")
    print("=" * 60)
    total_tr, total_va, total_te = 0, 0, 0
    for tclass in TIP_CLASSES:
        s = summary[tclass]
        print(f"  {tclass:<12}: {s['train']:4d} train, {s['val']:4d} val, {s['test']:3d} test")
        total_tr += s["train"]
        total_va += s["val"]
        total_te += s["test"]
    print("-" * 60)
    print(f"  Total: {total_tr} train, {total_va} val, {total_te} test tip crops")
    print("=" * 60)
    print("Ready for training: run 'python train_tip_classifier.py'")


if __name__ == "__main__":
    main()
