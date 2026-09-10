"""
Builds the folder structure Ultralytics YOLO classification training expects:

    yolo_dataset/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg
        test/<class_name>/*.jpg

Reads from your existing reference_images/ (split into train/val automatically)
and test_images/ (copied straight into test/, unchanged, since that's your
held-out set you never want touched by training).

Usage:  python prepare_dataset.py
"""

import os
import shutil
import random

REFERENCE_DIR = "reference_images"
TEST_DIR = "test_images"
OUTPUT_DIR = "yolo_dataset"
VAL_SPLIT = 0.2  # 20% of reference images held out for validation during training
SEED = 42

random.seed(SEED)


import cv2
from pen_cropper import crop_pen


def clear_and_make(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def process_and_save_files(files, src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        img = cv2.imread(src_path)
        if img is not None:
            cropped = crop_pen(img)
            cv2.imwrite(dst_path, cropped)
        else:
            shutil.copy2(src_path, dst_path)


def main():
    if not os.path.isdir(REFERENCE_DIR):
        print(f"ERROR: '{REFERENCE_DIR}' not found.")
        return

    clear_and_make(OUTPUT_DIR)

    class_names = sorted(
        d for d in os.listdir(REFERENCE_DIR)
        if os.path.isdir(os.path.join(REFERENCE_DIR, d))
    )

    if not class_names:
        print(f"ERROR: no class folders found inside '{REFERENCE_DIR}'.")
        return

    for class_name in class_names:
        src = os.path.join(REFERENCE_DIR, class_name)
        images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        # Separate video frames (already temporally split) from static photos
        vid_train = [f for f in images if f.startswith("vid_train_")]
        vid_val = [f for f in images if f.startswith("vid_val_")]
        static_photos = [f for f in images if not f.startswith("vid_")]

        # Split static photos randomly if present
        random.shuffle(static_photos)
        n_static_val = int(len(static_photos) * VAL_SPLIT) if static_photos else 0
        static_val = static_photos[:n_static_val]
        static_train = static_photos[n_static_val:]

        train_files = vid_train + static_train
        val_files = vid_val + static_val

        process_and_save_files(train_files, src, os.path.join(OUTPUT_DIR, "train", class_name))
        process_and_save_files(val_files, src, os.path.join(OUTPUT_DIR, "val", class_name))
        print(
            f"{class_name}: {len(train_files)} train ({len(vid_train)} vid + {len(static_train)} static), "
            f"{len(val_files)} val ({len(vid_val)} vid + {len(static_val)} static)"
        )

        test_src = os.path.join(TEST_DIR, class_name)
        if os.path.isdir(test_src):
            test_files = [f for f in os.listdir(test_src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            process_and_save_files(test_files, test_src, os.path.join(OUTPUT_DIR, "test", class_name))
            print(f"  + {len(test_files)} test (from {TEST_DIR})")
        else:
            print(f"  WARNING: no matching folder in '{TEST_DIR}' for {class_name}")

    print(f"\nDataset ready at '{OUTPUT_DIR}/'")
    print("Next step: run train_classifier.py")


if __name__ == "__main__":
    main()