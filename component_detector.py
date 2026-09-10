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
        Selects the farthest candidate contour from the grip along the pen axis
        to guarantee the rear back cap is accurately differentiated from the front top cap.
        """
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, np.array([90, 50, 40]), np.array([135, 255, 255]))
        cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > 60]
        if not valid:
            return None, False, (0, 0, w, h)

        valid.sort(key=cv2.contourArea, reverse=True)
        grip = valid[0]
        gx, gy, gw, gh = cv2.boundingRect(grip)
        gcx, gcy = gx + gw / 2.0, gy + gh / 2.0

        # Find backcap candidates: pick the candidate farthest from grip along pen body
        backcap = None
        cand_dists = []
        for cand in valid[1:]:
            bx, by, bw, bh = cv2.boundingRect(cand)
            d = np.hypot(gcx - (bx + bw / 2.0), gcy - (by + bh / 2.0))
            if d > 100:
                cand_dists.append((d, cand))

        if cand_dists:
            cand_dists.sort(key=lambda x: x[0], reverse=True)
            backcap = cand_dists[0][1]

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
        Analyzes an image using dual-expert mutual corroboration:
          - Expert 1: Whole-Pen Classifier (runs_classify/pen_states/weights/best.pt)
          - Expert 2: High-Resolution Tip Classifier (runs_classify/tip_classifier/weights/best.pt)
          - Physical Anchor: HSV Rear Back Cap Detector
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

        # 1. High-resolution tip classification
        tip_pred, tip_conf = None, 0.0
        if self.tip_model and tip_crop is not None and tip_crop.shape[0] > 20 and tip_crop.shape[1] > 20:
            res_tip = self.tip_model.predict(tip_crop, verbose=False)[0]
            tip_pred = res_tip.names[res_tip.probs.top1]
            tip_conf = float(res_tip.probs.top1conf)

        # 2. Whole pen classification (on focused pen crop)
        from pen_cropper import crop_pen
        pen_crop = crop_pen(img)
        pen_pred, pen_conf = None, 0.0
        if self.pen_model and pen_crop is not None:
            res_pen = self.pen_model.predict(pen_crop, verbose=False)[0]
            pen_pred = res_pen.names[res_pen.probs.top1]
            pen_conf = float(res_pen.probs.top1conf)

        # 3. DUAL-EXPERT MUTUAL CORROBORATION
        # ---------------------------------------------------------------------
        # Rule 1: Mutual Corroboration for State 4 (Top Cap Attached)
        # Prevents unilateral declaration of completion if the tip model still sees
        # the white cone cap, bare needle refill, or open barrel.
        if pen_pred == "state_4_topcap_on":
            if tip_pred == "topcap_tip":
                state = "state_4_topcap_on"
                conf = (pen_conf + tip_conf) / 2.0
            elif tip_pred == "conecap_tip":
                state = "state_3_conecap_on"
                conf = tip_conf
            elif tip_pred == "refill_tip":
                state = "state_2_refill_inserted"
                conf = tip_conf
            elif tip_pred == "empty_tip":
                state = "state_1_backcap_on" if has_backcap else "state_0_barrel_only"
                conf = tip_conf
            else:
                state = "state_4_topcap_on" if pen_conf >= 0.90 else "state_3_conecap_on"
                conf = pen_conf

        # Rule 2: Tip model strongly detects top cap
        elif tip_pred == "topcap_tip" and tip_conf >= 0.70:
            state = "state_4_topcap_on"
            conf = (pen_conf + tip_conf) / 2.0 if pen_pred == "state_4_topcap_on" else tip_conf

        # Rule 3: State 3 (Front Cone Cap Attached)
        elif tip_pred == "conecap_tip" and tip_conf >= 0.60:
            state = "state_3_conecap_on"
            conf = (pen_conf + tip_conf) / 2.0 if pen_pred == "state_3_conecap_on" else tip_conf

        elif pen_pred == "state_3_conecap_on":
            if tip_pred == "refill_tip" and tip_conf >= 0.70:
                state = "state_2_refill_inserted"
                conf = tip_conf
            elif tip_pred == "empty_tip" and tip_conf >= 0.80:
                state = "state_1_backcap_on" if has_backcap else "state_0_barrel_only"
                conf = tip_conf
            else:
                state = "state_3_conecap_on"
                conf = pen_conf

        # Rule 4: State 2 (Ink Refill Inserted)
        elif tip_pred == "refill_tip" and tip_conf >= 0.65 and (has_backcap or pen_pred == "state_2_refill_inserted"):
            state = "state_2_refill_inserted"
            conf = (pen_conf + tip_conf) / 2.0 if pen_pred == "state_2_refill_inserted" else tip_conf

        elif pen_pred == "state_2_refill_inserted":
            if tip_pred == "empty_tip" and tip_conf >= 0.85:
                state = "state_1_backcap_on" if has_backcap else "state_0_barrel_only"
                conf = tip_conf
            else:
                state = "state_2_refill_inserted"
                conf = pen_conf

        # Rule 5: States 0 and 1 (Barrel Body & Rear Back Cap)
        elif pen_pred in ["state_0_barrel_only", "state_1_backcap_on"]:
            if has_backcap:
                state = "state_1_backcap_on"
                conf = max(pen_conf, 0.75)
            else:
                state = "state_0_barrel_only"
                conf = max(pen_conf, 0.75)

        # Fallback
        else:
            state = pen_pred or "state_0_barrel_only"
            conf = pen_conf

        # Physical component presence
        has_barrel = True
        has_refill = (
            tip_pred in ["refill_tip", "conecap_tip", "topcap_tip"]
            or state in ["state_2_refill_inserted", "state_3_conecap_on", "state_4_topcap_on"]
        )
        has_conecap = (
            tip_pred in ["conecap_tip", "topcap_tip"]
            or state in ["state_3_conecap_on", "state_4_topcap_on"]
        )
        has_topcap = (
            (tip_pred == "topcap_tip" and state == "state_4_topcap_on")
            or (state == "state_4_topcap_on" and conf >= 0.85)
        )

        return {
            "predicted_state": state,
            "confidence": conf,
            "components": {
                "barrel": has_barrel,
                "back_cap": has_backcap,
                "refill": has_refill,
                "cone_cap": has_conecap,
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
