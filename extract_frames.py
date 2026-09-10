"""
Frame extraction script for training video datasets.

Features:
- Extracts video frames at a configurable sampling rate (default: 4 FPS).
- Filters out motion-blurred frames via Laplacian variance.
- Validates pen visibility via blue rubber grip detection (prevents saving off-screen or fully occluded frames).
- Option B Temporal Splitting: cleanly partitions frames by timestamp (e.g. first 80% for train, last 20% for val)
  to prevent train/val data leakage.
- Auto-detects videos in videos/ folder or accepts individual --video arguments.

Usage:
    python extract_frames.py                                                # Auto-extracts all videos in videos/
    python extract_frames.py --video videos/state0.mp4 --state state_0_barrel_only
"""

import os
import re
import argparse
import cv2
import numpy as np

REFERENCE_DIR = "reference_images"
DEFAULT_VIDEOS_DIR = "videos"

# Recognized states in sequential assembly order
VALID_STATES = [
    "state_0_barrel_only",
    "state_1_backcap_on",
    "state_2_refill_inserted",
    "state_3_conecap_on",
    "state_4_topcap_on",
]

# Patterns for auto-matching video filenames to state folders
STATE_PATTERNS = {
    "state_0_barrel_only": [r"state[_-]?0", r"barrel", r"^0\."],
    "state_1_backcap_on": [r"state[_-]?1", r"backcap", r"^1\."],
    "state_2_refill_inserted": [r"state[_-]?2", r"refill", r"^2\."],
    "state_3_conecap_on": [r"state[_-]?3", r"conecap", r"^3\."],
    "state_4_topcap_on": [r"state[_-]?4", r"topcap", r"^4\."],
}


def match_state_from_filename(filename):
    """Matches a filename like 'state0.mp4' or 'refill.mp4' to a valid state name."""
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    for state, patterns in STATE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, name):
                return state
    return None


def is_blurry(gray_frame, threshold=70.0):
    """Calculates Laplacian variance. Low variance indicates motion blur or lack of edge detail."""
    if threshold <= 0:
        return False
    var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return var < threshold


def has_pen_anchor(frame):
    """
    Checks if at least one blue component (grip/cap) is visible in the frame.
    Prevents saving frames where the pen is out of view or completely covered by a fist.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 40]), np.array([135, 255, 255]))
    cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if cv2.contourArea(c) > 60]
    return len(valid) > 0


def clean_existing_video_frames(target_dir):
    """Removes previously extracted vid_* frames from target directory while preserving static photos."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return 0

    removed = 0
    for f in os.listdir(target_dir):
        if f.startswith("vid_") and f.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(target_dir, f))
            removed += 1
    return removed


def extract_from_video(
    video_path,
    state_name,
    fps=4.0,
    blur_thresh=70.0,
    val_split=0.20,
    check_anchor=True,
    clean_old=True,
):
    """
    Extracts frames from a single video, filtering blur and applying temporal train/val split.
    """
    if not os.path.exists(video_path):
        print(f"  [ERROR] Video file not found: {video_path}")
        return None

    out_dir = os.path.join(REFERENCE_DIR, state_name)
    os.makedirs(out_dir, exist_ok=True)

    if clean_old:
        num_cleaned = clean_existing_video_frames(out_dir)
        if num_cleaned > 0:
            print(f"  [INFO] Cleaned {num_cleaned} old video frames from '{out_dir}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Could not open video: {video_path}")
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if total_frames > 0 else 0

    # Calculate sampling step
    step = max(1, int(round(video_fps / fps)))
    split_frame = int(total_frames * (1.0 - val_split))

    print(f"\nProcessing '{video_path}' -> {state_name}")
    print(f"  Video: {total_frames} frames ({duration_sec:.1f}s) @ {video_fps:.1f} fps")
    print(f"  Sampling rate: every {step} frames (~{video_fps / step:.1f} fps target)")
    print(f"  Option B Temporal Split: frames 0..{split_frame-1} -> TRAIN, {split_frame}..{total_frames} -> VAL")

    frame_idx = 0
    saved_train = 0
    saved_val = 0
    skipped_blur = 0
    skipped_anchor = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Blur Check
            if is_blurry(gray, blur_thresh):
                skipped_blur += 1
                frame_idx += 1
                continue

            # 2. Pen Anchor Check (ensures blue grip is visible)
            if check_anchor and not has_pen_anchor(frame):
                skipped_anchor += 1
                frame_idx += 1
                continue

            # 3. Temporal train/val assignment
            split_tag = "train" if frame_idx < split_frame else "val"
            fname = f"vid_{split_tag}_{frame_idx:06d}.jpg"
            save_path = os.path.join(out_dir, fname)
            cv2.imwrite(save_path, frame)

            if split_tag == "train":
                saved_train += 1
            else:
                saved_val += 1

        frame_idx += 1

    cap.release()

    print(f"  Results: {saved_train} train frames, {saved_val} val frames saved.")
    print(f"  Filtered out: {skipped_blur} blurry, {skipped_anchor} without visible grip.")
    return {
        "state": state_name,
        "train": saved_train,
        "val": saved_val,
        "blurry": skipped_blur,
        "no_anchor": skipped_anchor,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract video frames for pen assembly classifier.")
    parser.add_argument("--video", type=str, default=None, help="Path to single video file")
    parser.add_argument("--state", type=str, default=None, choices=VALID_STATES, help="Target state name")
    parser.add_argument("--videos_dir", type=str, default=DEFAULT_VIDEOS_DIR, help="Folder containing raw videos")
    parser.add_argument("--fps", type=float, default=4.0, help="Target extraction frames per second (default: 4.0)")
    parser.add_argument(
        "--blur_thresh",
        type=float,
        default=20.0,
        help="Laplacian variance threshold for blur filtering (default: 20.0, tuned for solid background)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.20,
        help="Fraction of video duration reserved for validation (Option B temporal split, default: 0.20)",
    )
    parser.add_argument(
        "--no_anchor_check",
        action="store_true",
        help="Disable blue grip visibility check",
    )
    parser.add_argument(
        "--no_clean",
        action="store_true",
        help="Do not remove old vid_* frames before extracting",
    )

    args = parser.parse_args()

    # Case 1: Single video mode
    if args.video:
        state = args.state or match_state_from_filename(args.video)
        if not state:
            print(f"ERROR: Could not infer state name from '{args.video}'. Please provide --state.")
            return
        extract_from_video(
            args.video,
            state,
            fps=args.fps,
            blur_thresh=args.blur_thresh,
            val_split=args.val_split,
            check_anchor=not args.no_anchor_check,
            clean_old=not args.no_clean,
        )
        return

    # Case 2: Batch directory mode
    if not os.path.exists(args.videos_dir):
        os.makedirs(args.videos_dir, exist_ok=True)
        print(f"Folder '{args.videos_dir}/' created.")
        print(f"Please drop your state videos (.mp4, .mov, etc.) into '{args.videos_dir}/' and run again.")
        return

    video_exts = (".mp4", ".mov", ".avi", ".mkv")
    candidates = [
        os.path.join(args.videos_dir, f)
        for f in os.listdir(args.videos_dir)
        if f.lower().endswith(video_exts)
    ]

    if not candidates:
        print(f"No video files found inside '{args.videos_dir}/'.")
        print("Please copy your videos into this folder, e.g.:")
        for st in VALID_STATES:
            print(f"  {args.videos_dir}/{st}.mp4  (or state0.mp4, state1.mp4, etc.)")
        return

    print(f"Found {len(candidates)} video file(s) in '{args.videos_dir}/'. Starting extraction...")
    results = []
    for vpath in candidates:
        state = match_state_from_filename(vpath)
        if not state:
            print(f"\n[WARNING] Skipping '{vpath}': Could not determine state name from filename.")
            print(f"          Name the file using one of: state0, state1, state2, state3, state4 (or full state name).")
            continue

        res = extract_from_video(
            vpath,
            state,
            fps=args.fps,
            blur_thresh=args.blur_thresh,
            val_split=args.val_split,
            check_anchor=not args.no_anchor_check,
            clean_old=not args.no_clean,
        )
        if res:
            results.append(res)

    print("\n" + "=" * 65)
    print(" EXTRACTION SUMMARY")
    print("=" * 65)
    total_tr, total_va = 0, 0
    for r in results:
        print(f"  {r['state']}: {r['train']} train, {r['val']} val (filtered {r['blurry']} blur, {r['no_anchor']} no-anchor)")
        total_tr += r["train"]
        total_va += r["val"]
    print("-" * 65)
    print(f"  Total: {total_tr} train frames, {total_va} val frames extracted.")
    print("=" * 65)
    print("\nNext step: Run 'python prepare_dataset.py' to crop pens and assemble 'yolo_dataset/'")


if __name__ == "__main__":
    main()
