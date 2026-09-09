"""
Pen Assembly Quality Inspection System
Main entrypoint for evaluating, running live demos, or training.

Usage:
    python main.py             # Interactive menu
    python main.py --eval      # Run benchmark evaluation
    python main.py --demo      # Start live camera inspection
    python main.py --train     # Retrain YOLO models
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Pen Assembly Quality Checker")
    parser.add_argument("--eval", action="store_true", help="Run benchmark evaluation on test dataset")
    parser.add_argument("--demo", action="store_true", help="Launch live webcam inspection HUD")
    parser.add_argument("--camera", type=str, default="0", help="Webcam device index (0, 1, 2) or DroidCam IP URL")
    parser.add_argument("--image", type=str, default=None, help="Inspect a single image file")
    parser.add_argument("--train", action="store_true", help="Retrain YOLO classification models")
    args = parser.parse_args()

    if args.eval:
        from evaluate import main as run_eval
        run_eval()
        return

    if args.demo:
        from live_demo import run_webcam
        from component_detector import ComponentDetector
        from state_machine import AssemblyStateMachine
        run_webcam(args.camera, ComponentDetector(), AssemblyStateMachine())
        return

    if args.image:
        from live_demo import run_image
        from component_detector import ComponentDetector
        from state_machine import AssemblyStateMachine
        run_image(args.image, ComponentDetector(), AssemblyStateMachine())
        return

    if args.train:
        from train_classifier import main as run_train
        run_train()
        return

    print("=" * 60)
    print("      PEN ASSEMBLY QUALITY INSPECTION SYSTEM")
    print("=" * 60)
    print("Select an option to run:")
    print("  [1] Run Benchmark Evaluation (evaluate.py)")
    print("  [2] Start Live Webcam Inspection (live_demo.py)")
    print("  [3] Retrain YOLO Models (train_classifier.py)")
    print("  [4] Re-generate Cropped Datasets (prepare_dataset.py)")
    print("  [q] Quit")
    print("=" * 60)

    try:
        choice = input("Enter choice (1-4 or q): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "1":
        from evaluate import main as run_eval
        run_eval()
    elif choice == "2":
        from live_demo import run_webcam
        from component_detector import ComponentDetector
        from state_machine import AssemblyStateMachine
        run_webcam(0, ComponentDetector(), AssemblyStateMachine())
    elif choice == "3":
        from train_classifier import main as run_train
        run_train()
    elif choice == "4":
        from prepare_dataset import main as run_prep
        run_prep()
    else:
        print("Exiting.")


if __name__ == "__main__":
    main()
