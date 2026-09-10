"""
Trains a YOLO classification model on your 5 pen-assembly states.

Usage:  python train_classifier.py
"""

import os
import torch
from ultralytics import YOLO


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")
    if device == "cpu":
        print(
            "WARNING: no GPU detected by torch. If you have one, check you installed "
            "the CUDA build of torch (see setup notes) rather than the CPU-only build."
        )

    # yolov8s-cls.pt (small) provides double the parameters of nano (~6.4M vs ~3.2M),
    # ideal for distinguishing fine-grained states (e.g. bare refill vs empty barrel)
    # when trained on ~1,000+ diverse video frames without overfitting.
    model = YOLO("yolov8s-cls.pt")

    model.train(
        data="yolo_dataset",   # folder with train/ and val/ subfolders from prepare_dataset.py
        epochs=80,             # patience below stops early once validation loss flattens
        imgsz=448,             # ideal resolution for cropped pen images
        batch=16,              # batch 16 provides stable gradient estimates with 1k+ images
        device=device,
        patience=25,           # allow more epochs for convergence
        erasing=0.0,           # disable random erasing so tiny pen parts aren't blacked out
        auto_augment=None,     # disable color jitter/solarization
        scale=0.1,             # avoid zooming out which makes the pen even smaller
        degrees=15.0,          # allow slight rotation
        fliplr=0.5,            # allow horizontal flip
        flipud=0.5,            # allow vertical flip
        optimizer="AdamW",     # better for fine-tuning on small datasets
        lr0=0.0005,            # tuned learning rate for fine-tuning
        project=os.path.abspath("runs_classify"),
        name="pen_states",
        exist_ok=True,
    )

    print("\nTraining complete.")
    print("Best weights saved to: runs_classify/pen_states/weights/best.pt")
    print("Next step: run evaluate.py to check accuracy on your held-out test set.")


if __name__ == "__main__":
    main()