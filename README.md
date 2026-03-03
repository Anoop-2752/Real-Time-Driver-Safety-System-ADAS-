# Real-Time Driver Safety System — ADAS

> An AI-powered co-pilot that monitors the driver and road simultaneously to prevent accidents before they happen.

Built with YOLOv8, MediaPipe, and OpenCV — deployable as a desktop dashboard or a live Streamlit web app.

---

## Features

| Module | What it does |
|---|---|
| **Lane Detection** | Detects left/right lane lines via Canny + Hough Transform, fills lane area green, triggers departure alert when vehicle drifts >15% off center |
| **Object Detection** | YOLOv8n detects cars, trucks, buses, motorcycles, bicycles, and pedestrians in real time |
| **Drowsiness Detection** | MediaPipe Face Mesh tracks 468 landmarks — computes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to detect drowsiness and yawning |
| **Collision Warning** | Estimates proximity by bounding-box-to-frame ratio — escalates SAFE → WARNING → DANGER with hysteresis to prevent flickering |
| **Audio Alerts** | Pygame plays distinct WAV alerts for lane departure, drowsiness, and collision danger |
| **Dual Dashboard** | Desktop (OpenCV window) and web (Streamlit) interfaces, both with real-time status panels and alert logs |

---

## Demo

[![Real-Time Driver Safety System Demo](https://img.youtube.com/vi/tohatVJFkyc/hqdefault.jpg)](https://www.youtube.com/watch?v=tohatVJFkyc)

> Click the thumbnail to watch the full demo on YouTube

```bash
# Run it yourself
python app.py                        # Desktop dashboard
streamlit run streamlit_app.py       # Web dashboard
```

The system accepts either **live webcam** input or **pre-recorded video files** for demo purposes (configured in `config.py`).

---

## How It Works

```
Front Camera (road view)
  │
  ├─► Lane Detection      →  Hough lines + EMA smoothing → departure check
  ├─► Object Detection    →  YOLOv8n inference → bounding boxes + labels
  └─► Collision Warning   →  bbox/frame ratio → SAFE / WARNING / DANGER
                                                      │
Driver Camera (face view)                             │
  │                                                   │
  └─► Drowsiness Detection → Face Mesh → EAR/MAR → drowsy/yawn alerts
                                                      │
                                           Dashboard + Audio Alerts
```

Both camera pipelines run in **parallel threads** using `concurrent.futures.ThreadPoolExecutor` to maximize frame rate.

**Temporal smoothing** (Exponential Moving Average, α = 0.3) is applied to lane lines and EAR/MAR values to eliminate single-frame noise and jittery alerts.

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Object Detection | [YOLOv8n](https://github.com/ultralytics/ultralytics) (ultralytics) |
| Face / Landmark | [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh) |
| Computer Vision | OpenCV 4.9 |
| Deep Learning | PyTorch 2.2 |
| Web Dashboard | Streamlit 1.35 |
| Audio | Pygame 2.5 |
| Testing | pytest |

---

## Project Structure

```
Real-Time-Driver-Safety-System-ADAS-/
│
├── app.py                      # Desktop dashboard (OpenCV window)
├── streamlit_app.py            # Web dashboard (Streamlit)
├── config.py                   # All tuneable parameters in one place
├── generate_sounds.py          # Utility to regenerate alert WAV files
├── requirements.txt
│
├── modules/
│   ├── lane_detection.py       # Hough Transform + EMA lane tracking
│   ├── object_detection.py     # YOLOv8 wrapper
│   ├── drowsiness_detection.py # MediaPipe EAR/MAR computation
│   └── collision_warning.py    # Proximity state machine
│
├── models/
│   └── yolov8n.pt              # YOLOv8 nano weights (auto-downloaded)
│
├── assets/
│   ├── sounds/
│   │   ├── lane_alert.wav
│   │   ├── drowsy_alert.wav
│   │   └── collision_alert.wav
│   └── test_videos/
│       ├── dashcam.mp4         # Road / front-view demo video
│       └── facecam.mp4         # Driver face demo video
│
└── tests/
    ├── __init__.py
    └── test_modules.py         # 20 pytest unit tests
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Real-Time-Driver-Safety-System-ADAS-.git
cd Real-Time-Driver-Safety-System-ADAS-
```

### 2. Create a conda environment (recommended)

```bash
conda create -n adas python=3.10 -y
conda activate adas
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. YOLOv8 weights

The model file `models/yolov8n.pt` is included. If missing, it will be auto-downloaded by ultralytics on first run.

---

## Demo Videos Setup

To run in demo mode (no webcam required), place two video files in `assets/test_videos/`:

| File | Content |
|---|---|
| `dashcam.mp4` | Road / front-view dashcam footage |
| `facecam.mp4` | Driver face footage |

**Recommended source:** [BDD100K Dataset](https://www.bdd100k.com/) — download any dashcam clip from the 100K videos set. For the face camera, any front-facing driver recording works.

Both videos loop automatically when they reach the end.

---

## Usage

### Desktop Dashboard (`app.py`)

Full dual-panel OpenCV window with front camera, driver monitor, and system status panel.

```bash
python app.py
```

Press **Q** to quit.

### Web Dashboard (`streamlit_app.py`)

Streamlit interface with live video feed, status cards, metrics, and alert log. Runs in the browser.

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser, then click **Start System**.

---

## Configuration

All parameters are in [config.py](config.py). Key settings:

```python
# ── Demo vs Live ────────────────────────────────────
USE_VIDEO    = True                          # True = video files, False = live webcam
VIDEO_FRONT  = "assets/test_videos/dashcam.mp4"
VIDEO_DRIVER = "assets/test_videos/facecam.mp4"
CAMERA_INDEX = 0                             # Webcam index (used when USE_VIDEO = False)

# ── Drowsiness ───────────────────────────────────────
EAR_THRESHOLD      = 0.25   # Below this → eyes closing
MAR_THRESHOLD      = 0.6    # Above this → yawning
DROWSY_FRAME_COUNT = 20     # Consecutive frames before drowsiness alert
YAWN_FRAME_COUNT   = 10     # Consecutive frames before yawn alert

# ── Collision ────────────────────────────────────────
COLLISION_BOX_THRESHOLD = 0.35  # bbox > 35% of frame area → DANGER zone

# ── Smoothing ────────────────────────────────────────
SMOOTHING_ALPHA = 0.3   # EMA weight: 0.3 = 30% new, 70% history

# ── Lane Detection ───────────────────────────────────
ROI_HEIGHT_RATIO = 0.6  # Region of interest starts at 60% of frame height

# ── Object Detection ─────────────────────────────────
YOLO_CONFIDENCE      = 0.5
DETECTION_CLASSES    = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorbike, bus, truck
```

---

## Running Tests

```bash
# Activate the adas environment first
conda activate adas

pytest tests/ -v
```

Expected output: **20 passed**.

The test suite covers:
- Lane departure logic (centered, off-center, missing lines)
- EMA smoothing convergence
- EAR and MAR formula correctness
- MAR divide-by-zero safety
- Drowsiness and yawn frame-counter hysteresis
- Collision box area calculation
- SAFE / WARNING / DANGER state transitions
- State hold during frame buildup (no flickering)
- Closest-vehicle selection

---

## Alert Logic

### Drowsiness (EAR)
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)

EAR < 0.25  for 20 consecutive frames  →  DROWSY alert
MAR > 0.60  for 10 consecutive frames  →  YAWN alert
```

### Collision Warning
```
Proximity ratio = vehicle bounding box area / frame area

ratio < 15%                            →  SAFE
ratio 15–35%  for 5+ frames            →  WARNING
ratio > 35%   for 5+ frames            →  DANGER
```

State changes require 5 consecutive confirming frames to prevent false triggers.

### Lane Departure
```
Lane center = (left_lane_x + right_lane_x) / 2
Deviation   = |lane_center - frame_center|

Deviation > 15% of frame width  →  LANE DEPARTURE alert
```

---

## Architecture Notes

- **Parallel processing:** front and driver pipelines run in separate threads via `ThreadPoolExecutor` — reduces effective latency compared to serial processing.
- **EMA smoothing:** applied to lane line coordinates and EAR/MAR values — eliminates single-frame noise without adding significant delay (α = 0.3).
- **Hysteresis:** collision and drowsiness alerts require N consecutive confirming frames before triggering — prevents jitter and false positives.
- **Auto-loop:** demo video files loop automatically when they reach the end.
- **Resource safety:** camera handles are stored in `st.session_state` (Streamlit) and released in `try/finally` blocks to prevent leaks.

---

## Built With

- **Author:** Anoop Krishna
- **Purpose:** ADAS  Project
- **Stack:** YOLOv8 · MediaPipe · OpenCV · Streamlit · PyTorch
