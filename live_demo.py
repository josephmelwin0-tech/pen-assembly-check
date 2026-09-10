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
    Draws an industrial assembly quality inspection HUD onto the frame.
    Strictly enforces sequential assembly order, highlights skipped steps in RED,
    and displays verified progress checklists.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    state = result.get("predicted_state", "unknown")
    conf = result.get("confidence", 0.0)

    if smoothed:
        status, detail, consensus_state, votes_ratio = state_machine.update_smoothed(state)
        display_state = consensus_state if consensus_state else state
    else:
        status, detail = state_machine.update(state)
        votes_ratio = "1/1"
        display_state = state

    is_error = status == "error" or state_machine.error_active
    is_done = state_machine.is_complete()

    # Determine header banner color and border
    if is_error:
        header_color = (20, 20, 180)   # Crimson Red
        border_color = (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (w, h), border_color, 4)
    elif is_done:
        header_color = (20, 130, 20)   # Forest Green
        border_color = (0, 220, 0)
        cv2.rectangle(frame, (0, 0), (w, h), border_color, 4)
    elif status == "advanced":
        header_color = (0, 130, 180)   # Amber / Gold
    else:
        header_color = (25, 25, 25)    # Industrial Dark Gray

    # Draw semi-transparent header bar (taller for readability)
    header_h = 145
    cv2.rectangle(overlay, (0, 0), (w, header_h), header_color, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Calculate horizontal layout split
    panel_w = 230 if w >= 550 else 180
    start_x = max(w - panel_w, int(w * 0.55))

    # --- Header Information (Left Side) ---
    max_text_w = start_x - 15
    if is_error:
        cv2.putText(frame, "SEQUENCE REJECTED!", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        err_msg = detail or state_machine.error_detail or "Sequence error detected"
        # Truncate if very long
        if len(err_msg) > 38 and w < 700:
            err_msg = err_msg[:35] + "..."
        cv2.putText(frame, err_msg, (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 230, 255), 1)
        cv2.putText(frame, f"Detected: {display_state.upper()} ({conf*100:.0f}%)", (15, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(frame, "Fix missing part or press 'r' to reset", (15, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
    elif is_done:
        cv2.putText(frame, "100% COMPLETE & VERIFIED!", (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, "All 5 steps correctly assembled in order.", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 255, 220), 1)
        cv2.putText(frame, "READY FOR NEXT PEN - Press 'r' to reset", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    else:
        cur_title = state_machine.get_step_title(state_machine.current_index)
        cv2.putText(frame, f"INSPECTION: {cur_title.upper()}", (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(frame, f"LIVE: {display_state.upper()} ({conf*100:.0f}%)", (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        next_step = state_machine.get_step_title(state_machine.current_index + 1)
        stat_color = (0, 215, 255) if status == "advanced" else (220, 180, 50)
        cv2.putText(frame, f"STATUS: {status.upper()} (Dwell: {votes_ratio})", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, stat_color, 1)
        cv2.putText(frame, f"Next required: [{next_step}]", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    # --- Sequential Assembly Pipeline Checklist (Right Side) ---
    # Draw right panel semi-transparent container
    cv2.rectangle(overlay, (start_x - 10, 5), (w - 5, header_h - 5), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    steps = state_machine.get_steps_for_hud()
    for i, s in enumerate(steps):
        st = s["status"]
        if st == "verified":
            tag = "[PASS] "
            col = (60, 235, 60)   # Green
        elif st == "skipped":
            tag = "[MISS] "
            col = (40, 40, 255)   # Bright Red
        elif st == "current":
            tag = "[NOW ] "
            col = (240, 210, 40)  # Cyan
        elif st == "error_current":
            tag = "[BLCK] "
            col = (50, 150, 255)  # Orange
        else:
            tag = "[    ] "
            col = (140, 140, 140) # Dim Gray

        # Shorten step label for compact display
        title = s["title"].replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "").replace("5. ", "")
        text = f"{tag}{i+1}.{title}"
        cv2.putText(frame, text, (start_x, 26 + i * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2 if st in ["verified", "skipped"] else 1)

    # Highlight tip box if available
    tip_box = result.get("tip_box")
    if tip_box:
        tx, ty, tw, th = tip_box
        sig = result.get("signals", {})
        tip_lbl = sig.get("tip_pred")
        tip_c = sig.get("tip_conf", 0.0)
        label_text = f"TIP: {tip_lbl} ({tip_c*100:.0f}%)" if tip_lbl else "TIP ROI"
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 255), 2)
        cv2.putText(frame, label_text, (tx, max(ty - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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


def run_webcam(camera_source, detector, sm):
    # If camera_source is an integer or numeric string ("0", "1"), convert to int
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera source: {camera_source}")
        print("Tip: If using DroidCam client, try device index 1 or 2:")
        print("     python live_demo.py --camera 1")
        print("     Or use the direct WiFi IP URL from the DroidCam phone app:")
        print("     python live_demo.py --camera http://<PHONE_IP>:4747/video")
        return

    print(f"Live assembly checker started on source [{camera_source}].")
    print("Press 'q' to quit, 'r' to reset sequence.")
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
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Webcam device index (0, 1, 2) or DroidCam IP URL (e.g. http://192.168.1.5:4747/video)",
    )
    args = parser.parse_args()

    detector = ComponentDetector()
    sm = AssemblyStateMachine()

    if args.image:
        run_image(args.image, detector, sm)
    else:
        run_webcam(args.camera, detector, sm)


if __name__ == "__main__":
    main()
