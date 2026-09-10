"""
Trains the localized Tip Classifier on yolo_tip_dataset/ using the NVIDIA GPU.

Usage:
    python train_tip_classifier.py
"""

import os
import torch
from ultralytics import YOLO


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training Tip Classifier on device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # yolov8s-cls provides high capacity for fine-grained tip textures
    model = YOLO("yolov8s-cls.pt")

    model.train(
        data="yolo_tip_dataset",
        epochs=60,
        imgsz=192,
        batch=16,
        device=device,
        patience=15,
        erasing=0.0,
        auto_augment=None,
        degrees=10.0,
        fliplr=0.5,
        flipud=0.5,
        optimizer="AdamW",
        lr0=0.0005,
        project=os.path.abspath("runs_classify"),
        name="tip_classifier",
        exist_ok=True,
    )

    print("\nTip Classifier training complete.")
    print("Best weights saved to: runs_classify/tip_classifier/weights/best.pt")


if __name__ == "__main__":
    main()
