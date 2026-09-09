# Pen Assembly Quality Inspection System

An automated Computer Vision & Deep Learning inspection system for manufacturing assembly lines. It tracks and verifies the 5 sequential stages of pen assembly in real time, detecting missing components and out-of-order steps.

---

## Assembly States

| State | Name | Description | Key Components |
| :---: | :--- | :--- | :--- |
| **0** | `state_0_barrel_only` | Bare pen body | Barrel only, empty ends |
| **1** | `state_1_backcap_on` | Rear plug attached | White back cap inserted |
| **2** | `state_2_refill_inserted`| Ink cartridge inserted | Bare 0.7mm metallic needle tip exposed |
| **3** | `state_3_conecap_on` | Front cone screwed on | Conical metallic cone cap attached |
| **4** | `state_4_topcap_on` | Finished pen | Blue top cap covering the nib |

---

## Accuracy Progression

| Approach | Accuracy on Test Set |
| :--- | :---: |
| Baseline Raw YOLOv8n-cls (224px) | **24.0%** (6/25) |
| Option 2: High-Res 640px + Augmentation Tuning | **36.0%** (9/25) |
| Option 1: Autonomous Pen Cropper (HSV ROI Anchor) | **48.0%** (12/25) |
| **Hierarchical Dual-Classifier + Component Logic** | **76.0%** (19/25) |
| **Live Stream with Temporal Rolling Consensus** | **100% Sequence Order** |

---

## Quick Setup (For Windows / macOS / Linux)

### 1. Prerequisites
- Python 3.10 to 3.12 (or 3.14)
- Git
- A webcam (for live mode)

### 2. Clone the Repository
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/pen-assembly-checker.git
cd pen-assembly-checker
```

### 3. Create a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Option A: Interactive Menu
```bash
python main.py
```

### Option B: Run Benchmark Evaluation
Evaluates both the raw whole-pen classifier and the hierarchical component checker against the held-out test dataset:
```bash
python evaluate.py
# or:
python main.py --eval
```

### Option C: Live Inspection HUD (Webcam or DroidCam Phone Camera)
Launches the real-time inspection interface with component checklist and temporal consensus smoothing:
```bash
# Using default built-in laptop webcam (index 0):
python live_demo.py

# Using DroidCam via USB/PC Client (index 1 or 2):
python live_demo.py --camera 1

# Using DroidCam via direct WiFi IP (no PC client needed):
python live_demo.py --camera http://<PHONE_IP>:4747/video
```
- Press **`r`** to reset the assembly sequence back to Step 0.
- Press **`q`** to quit.

### Option D: Inspect a Single Image
```bash
python live_demo.py --image "test_images/state_2_refill_inserted/IMG-20260909-WA0203 (1).jpg"
```
The annotated result with bounding boxes and state readout will be saved to `demo_output.jpg`.

---

## How to Retrain / Add New Photos

If you collect new images (15–20 images per state recommended):

1. Put your new photos into the corresponding folder under `reference_images/`:
   - `reference_images/state_0_barrel_only/`
   - `reference_images/state_1_backcap_on/`
   - `reference_images/state_2_refill_inserted/`
   - `reference_images/state_3_conecap_on/`
   - `reference_images/state_4_topcap_on/`
2. Re-crop and prepare dataset splits:
   ```bash
   python prepare_dataset.py
   ```
3. Train the YOLO model:
   ```bash
   python train_classifier.py
   ```
4. Verify new accuracy:
   ```bash
   python evaluate.py
   ```

---

## Architecture Overview

```
Raw Camera Frame (1280x960)
         │
         ▼
┌──────────────────┐
│  pen_cropper.py  │  ──> Detects blue rubber grip anchor via HSV segmentation
└──────────────────┘  ──> Extracts ~500px focused Pen Crop + 180px Tip Crop
         │
         ├─────────────────────────────────────────┐
         ▼                                         ▼
┌─────────────────────────┐             ┌─────────────────────────┐
│  Whole-Pen Classifier   │             │   Tip ROI Classifier    │
│  (YOLOv8n-cls @ 448px)  │             │  (YOLOv8n-cls @ 180px)  │
└─────────────────────────┘             └─────────────────────────┘
         │                                         │
         └────────────────────┬────────────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │   component_detector.py  │
                 │   (Hierarchical Fusion)  │
                 └──────────────────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     state_machine.py     │
                 │  (7-Frame Consensus &   │
                 │   Physical Rules Check)  │
                 └──────────────────────────┘
                              │
                              ▼
                    PASS / HOLD / ERROR HUD
```
