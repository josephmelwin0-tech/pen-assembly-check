import cv2
import numpy as np


def crop_pen(img, pad=170):
    """
    Detects the pen's blue components (grip, backcap, topcap) and crops
    a focused region-of-interest around the pen so it fills the frame.
    Works seamlessly on both desk shots and hand-held assembly shots.
    Generous padding guarantees delicate parts (0.7mm refill tip, cone nozzle,
    backcap, threads) are never accidentally clipped.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Blue mask for grip / backcap / topcap
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 40]), np.array([135, 255, 255]))
    cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if cv2.contourArea(c) > 80]

    if not valid:
        return img  # fallback to original if no blue detected

    all_pts = np.concatenate([c.reshape(-1, 2) for c in valid])
    bx, by, bw, bh = cv2.boundingRect(all_pts)

    # State 0 (empty barrel): only the grip is blue, so bounding box is small (~90px).
    # The clear barrel extends ~440px from the grip center.
    if max(bw, bh) < 200:
        p = 440
        cx, cy = bx + bw // 2, by + bh // 2
        x1 = max(0, cx - p)
        y1 = max(0, cy - p)
        x2 = min(w, cx + p)
        y2 = min(h, cy + p)
    else:
        # States 1-4: both ends or topcap have blue components.
        # pad=170 ensures the protruding cone cap / refill tip is fully included.
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)

    return img[y1:y2, x1:x2]
