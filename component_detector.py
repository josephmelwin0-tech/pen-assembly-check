import os
import cv2
import numpy as np
from ultralytics import YOLO

PEN_MODEL_PATH = "runs_classify/pen_states/weights/best.pt"
TIP_MODEL_PATH = "runs_classify/tip_classifier/weights/best.pt"


class ComponentDetector:
    def __init__(self, pen_model_path=PEN_MODEL_PATH, tip_model_path=TIP_MODEL_PATH):
        self.pen_model = YOLO(pen_model_path) if os.path.exists(pen_model_path) else None
        self.tip_model = YOLO(tip_model_path) if os.path.exists(tip_model_path) else None

    def extract_tip_and_endpoints(self, img):
        """
        Locates the blue grip, backcap, and computes the tip bounding box.
        """
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([90, 50, 40]), np.array([135, 255, 255]))
        cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > 120]
        if not valid:
            return None, False, (0, 0, w, h)

        valid.sort(key=cv2.contourArea, reverse=True)
        grip = valid[0]
        gx, gy, gw, gh = cv2.boundingRect(grip)
        gcx, gcy = gx + gw / 2.0, gy + gh / 2.0

        backcap = None
        for cand in valid[1:]:
            bx, by, bw, bh = cv2.boundingRect(cand)
            if np.hypot(gcx - (bx + bw / 2.0), gcy - (by + bh / 2.0)) > 120:
                backcap = cand
                break

        has_backcap = backcap is not None

        if has_backcap:
            bx, by, bw, bh = cv2.boundingRect(backcap)
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            dx, dy = gcx - bcx, gcy - bcy
            dist = np.hypot(dx, dy)
            ux, uy = dx / dist, dy / dist
            tip_cx = int(gcx + ux * (max(gw, gh) * 0.75))
            tip_cy = int(gcy + uy * (max(gw, gh) * 0.75))
        else:
            tip_cx, tip_cy = int(gcx), int(gcy)

        rad = 90
        x1, y1 = max(0, tip_cx - rad), max(0, tip_cy - rad)
        x2, y2 = min(w, tip_cx + rad), min(h, tip_cy + rad)
        tip_crop = img[y1:y2, x1:x2]

        return tip_crop, has_backcap, (x1, y1, x2 - x1, y2 - y1)

    def analyze(self, img_or_path):
        """
        Analyzes an image and returns component breakdown and state prediction.
        """
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path

        if img is None:
            return {
                "error": "Failed to load image",
                "predicted_state": "unknown",
                "confidence": 0.0,
            }

        tip_crop, has_backcap, tip_box = self.extract_tip_and_endpoints(img)

        # Tip classification
        tip_pred, tip_conf = None, 0.0
        if self.tip_model and tip_crop is not None and tip_crop.shape[0] > 20 and tip_crop.shape[1] > 20:
            res_tip = self.tip_model.predict(tip_crop, verbose=False)[0]
            tip_pred = res_tip.names[res_tip.probs.top1]
            tip_conf = float(res_tip.probs.top1conf)

        # Whole pen classification (on focused pen crop)
        from pen_cropper import crop_pen
        pen_crop = crop_pen(img)
        pen_pred, pen_conf = None, 0.0
        if self.pen_model and pen_crop is not None:
            res_pen = self.pen_model.predict(pen_crop, verbose=False)[0]
            pen_pred = res_pen.names[res_pen.probs.top1]
            pen_conf = float(res_pen.probs.top1conf)

        # State 4: Top cap on
        if pen_pred == "state_4_topcap_on" and pen_conf > 0.4:
            state = "state_4_topcap_on"
            conf = pen_conf
        elif tip_pred == "topcap_tip" and tip_conf > 0.5:
            state = "state_4_topcap_on"
            conf = tip_conf
        # State 3: Cone cap on
        elif tip_pred == "conecap_tip" and tip_conf > 0.55:
            state = "state_3_conecap_on"
            conf = tip_conf
        elif pen_pred == "state_3_conecap_on" and pen_conf > 0.6:
            state = "state_3_conecap_on"
            conf = pen_conf
        # State 0 vs 1: Empty tip
        elif tip_pred == "empty_tip" and tip_conf > 0.5:
            if has_backcap:
                state = "state_1_backcap_on"
                conf = tip_conf
            else:
                state = "state_0_barrel_only"
                conf = tip_conf
        # State 2: Refill inserted
        elif tip_pred == "refill_tip" and tip_conf > 0.45:
            if not has_backcap and pen_pred == "state_0_barrel_only" and pen_conf > 0.7:
                state = "state_0_barrel_only"
                conf = pen_conf
            else:
                state = "state_2_refill_inserted"
                conf = tip_conf
        elif not has_backcap:
            state = "state_0_barrel_only"
            conf = 0.85
        else:
            state = "state_1_backcap_on"
            conf = 0.85

        has_barrel = True
        has_refill = state in ["state_2_refill_inserted", "state_3_conecap_on", "state_4_topcap_on"]
        has_conecap = state in ["state_3_conecap_on", "state_4_topcap_on"]
        has_topcap = state == "state_4_topcap_on"

        return {
            "predicted_state": state,
            "confidence": conf,
            "components": {
                "barrel": has_barrel,
                "back_cap": has_backcap,
                "refill": has_refill or has_conecap or has_topcap,
                "cone_cap": has_conecap or has_topcap,
                "top_cap": has_topcap,
            },
            "signals": {
                "tip_pred": tip_pred,
                "tip_conf": tip_conf,
                "pen_pred": pen_pred,
                "pen_conf": pen_conf,
                "has_backcap": has_backcap,
            },
            "tip_box": tip_box,
        }
