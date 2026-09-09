"""
Evaluates both the YOLO classifier and the hierarchical component-based
assembly checker on your held-out test set.

Usage:  python evaluate.py
"""

import os
from ultralytics import YOLO
from component_detector import ComponentDetector
from state_machine import AssemblyStateMachine

PEN_MODEL_PATH = "runs_classify/pen_states/weights/best.pt"
TEST_DIR = "yolo_dataset/test"
RAW_TEST_DIR = "test_images"


def evaluate_classifier(model_path, test_dir):
    print("=" * 60)
    print(" 1. WHOLE-PEN YOLO CLASSIFIER EVALUATION")
    print("=" * 60)
    if not os.path.exists(model_path):
        print(f"ERROR: model not found at '{model_path}'. Run train_classifier.py first.")
        return 0, 0

    model = YOLO(model_path)
    correct, total = 0, 0
    confused = []

    for class_name in sorted(os.listdir(test_dir)):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            fpath = os.path.join(class_dir, fname)
            result = model.predict(fpath, verbose=False)[0]
            predicted_class = result.names[result.probs.top1]
            confidence = float(result.probs.top1conf)

            is_correct = predicted_class == class_name
            correct += int(is_correct)
            total += 1

            mark = "correct" if is_correct else "WRONG"
            print(f"[{mark}] {class_name}/{fname} -> predicted={predicted_class}, confidence={confidence:.3f}")

            if not is_correct:
                confused.append((class_name, predicted_class, confidence))

    acc = 100 * correct / total if total > 0 else 0
    print(f"\nClassifier Accuracy: {correct}/{total} ({acc:.1f}%)\n")
    return correct, total


def evaluate_hierarchical(raw_test_dir):
    print("=" * 60)
    print(" 2. HIERARCHICAL COMPONENT CHECKER EVALUATION")
    print("=" * 60)
    if not os.path.isdir(raw_test_dir):
        print(f"ERROR: '{raw_test_dir}' not found.")
        return 0, 0

    detector = ComponentDetector()
    sm = AssemblyStateMachine()
    correct, total = 0, 0
    confused = []

    for class_name in sorted(os.listdir(raw_test_dir)):
        class_dir = os.path.join(raw_test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            fpath = os.path.join(class_dir, fname)
            res = detector.analyze(fpath)
            predicted_class = res["predicted_state"]
            confidence = res["confidence"]
            comps = res.get("components", {})

            # Validate physical consistency
            is_valid, msg = sm.verify_components(comps)

            is_correct = predicted_class == class_name
            correct += int(is_correct)
            total += 1

            mark = "correct" if is_correct else "WRONG"
            print(f"[{mark}] {class_name}/{fname} -> predicted={predicted_class} (conf={confidence:.2f})")

            if not is_correct:
                confused.append((class_name, predicted_class, confidence))

    acc = 100 * correct / total if total > 0 else 0
    print(f"\nHierarchical Checker Accuracy: {correct}/{total} ({acc:.1f}%)\n")
    return correct, total


def main():
    c_corr, c_tot = evaluate_classifier(PEN_MODEL_PATH, TEST_DIR)
    h_corr, h_tot = evaluate_hierarchical(RAW_TEST_DIR)

    print("=" * 60)
    print(" ACCURACY COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Baseline YOLOv8n-cls (original):           24.0% (6/25)")
    print(f"  Cropped YOLOv8n-cls (Option 1):            {100*c_corr/c_tot:.1f}% ({c_corr}/{c_tot})")
    print(f"  Hierarchical Component Checker (Option C): {100*h_corr/h_tot:.1f}% ({h_corr}/{h_tot})")
    print("=" * 60)


if __name__ == "__main__":
    main()