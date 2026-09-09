"""
Live visual assembly checker for pen assembly.
Runs on either a live camera feed or an input test image.

Usage:
    python live_demo.py                       # Opens default webcam
    python live_demo.py --image <image_path>  # Evaluates a single image with visual overlay
"""

import argparse
import os
import cv2
from component_detector import ComponentDetector
from state_machine import AssemblyStateMachine


def draw_overlay(frame, result, state_machine, smoothed=True):
    """
    Draws a clean industrial inspection HUD onto the frame.
    Supports temporal smoothing for video streams to prevent flicker.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Draw semi-transparent header bar
    cv2.rectangle(overlay, (0, 0), (w, 140), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    state = result.get("predicted_state", "unknown")
    conf = result.get("confidence", 0.0)
    comps = result.get("components", {})

    if smoothed:
        status, detail, consensus_state, votes_ratio = state_machine.update_smoothed(state)
        display_state = consensus_state if consensus_state else state
    else:
        status, detail = state_machine.update(state)
        votes_ratio = "1/1"
        display_state = state

    # Color coding
    if status == "error":
        status_color = (50, 50, 220)   # Red
    elif state_machine.is_complete():
        status_color = (60, 220, 60)   # Green
    elif status == "advanced":
        status_color = (0, 215, 255)   # Gold
    else:
        status_color = (220, 180, 50)  # Cyan/Blue

    # Status Header
    cv2.putText(frame, f"STATE: {display_state.upper()}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(frame, f"CONF: {conf*100:.1f}% | STABILITY: {votes_ratio}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    status_text = f"STATUS: {status.upper()}"
    if detail and detail != "stabilizing":
        status_text += f" ({detail})"
    elif detail == "stabilizing":
        status_text += " (STABILIZING...)"
    cv2.putText(frame, status_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Component Checklist on the right
    comp_names = [
        ("Barrel", comps.get("barrel", False)),
        ("Back Cap", comps.get("back_cap", False)),
        ("Refill", comps.get("refill", False)),
        ("Cone Cap", comps.get("cone_cap", False)),
        ("Top Cap", comps.get("top_cap", False)),
    ]

    start_x = max(w - 320, 300)
    for i, (name, present) in enumerate(comp_names):
        check_mark = "[x]" if present else "[ ]"
        col = (60, 230, 60) if present else (120, 120, 120)
        cv2.putText(frame, f"{check_mark} {name}", (start_x, 30 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # Highlight tip box if available
    tip_box = result.get("tip_box")
    if tip_box:
        tx, ty, tw, th = tip_box
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 255), 2)
        cv2.putText(frame, "TIP ROI", (tx, max(ty - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return frame


def run_image(image_path, detector, sm):
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return
    img = cv2.imread(image_path)
    res = detector.analyze(img)
    annotated = draw_overlay(img, res, sm, smoothed=False)

    out_path = "demo_output.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\nAssembly State: {res['predicted_state']} (Confidence: {res['confidence']:.2f})")
    print(f"Components: {res.get('components', {})}")
    print(f"Annotated frame saved to '{out_path}'")


def run_webcam(camera_idx, detector, sm):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_idx}")
        return

    print("Live assembly checker started. Press 'q' to quit, 'r' to reset sequence.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = detector.analyze(frame)
        annotated = draw_overlay(frame, res, sm, smoothed=True)
        cv2.imshow("Pen Assembly Checker (Live)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            sm.reset()
            print("Assembly state machine reset.")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live Pen Assembly Inspection")
    parser.add_argument("--image", type=str, default=None, help="Path to single test image")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0)")
    args = parser.parse_args()

    detector = ComponentDetector()
    sm = AssemblyStateMachine()

    if args.image:
        run_image(args.image, detector, sm)
    else:
        run_webcam(args.camera, detector, sm)


if __name__ == "__main__":
    main()
