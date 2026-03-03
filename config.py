# config.py

# ─── Camera / Video ───────────────────────────────
CAMERA_INDEX = 0                  # 0 = default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# ─── Lane Detection ───────────────────────────────
LANE_COLOR = (0, 255, 0)          # Green
LANE_THICKNESS = 3
ROI_HEIGHT_RATIO = 0.6            # Region of interest starts at 60% of frame height

# ─── Object Detection ─────────────────────────────
YOLO_MODEL_PATH = "models/yolov8n.pt"
YOLO_CONFIDENCE = 0.5             # Minimum confidence threshold
DETECTION_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorbike, bus, truck

# ─── Drowsiness Detection ─────────────────────────
EAR_THRESHOLD = 0.25              # Eye Aspect Ratio — below this = drowsy
MAR_THRESHOLD = 0.6               # Mouth Aspect Ratio — above this = yawning
DROWSY_FRAME_COUNT = 20           # Consecutive frames before alert triggers

# ─── Collision Warning ────────────────────────────
COLLISION_BOX_THRESHOLD = 0.35    # If bounding box occupies 35% of frame = danger
COLLISION_COLOR_SAFE = (0, 255, 0)
COLLISION_COLOR_WARNING = (0, 165, 255)
COLLISION_COLOR_DANGER = (0, 0, 255)

# ─── Alert Sounds ─────────────────────────────────
SOUND_LANE_ALERT = "assets/sounds/lane_alert.mp3"
SOUND_DROWSY_ALERT = "assets/sounds/drowsy_alert.mp3"
SOUND_COLLISION_ALERT = "assets/sounds/collision_alert.mp3"

# ─── UI Colors (BGR format) ───────────────────────
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_ORANGE = (0, 165, 255)