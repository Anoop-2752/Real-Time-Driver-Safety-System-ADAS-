# tests/test_modules.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ─── Lane Detection Tests ──────────────────────────────────────

class TestLaneDetector:
    def setup_method(self):
        from modules.lane_detection import LaneDetector
        self.detector = LaneDetector()

    def test_departure_centered_lanes(self):
        """No alert when vehicle is centered between lanes."""
        width = 640
        # Left lane at x=160, right lane at x=480 → center=320 = frame center
        left_line  = (160, 480, 200, 288)
        right_line = (480, 480, 440, 288)
        assert self.detector._check_departure(left_line, right_line, width) is False

    def test_departure_off_center(self):
        """Alert when lane center deviates >15% from frame center."""
        width = 640
        # Both lanes shifted far right → vehicle appears left of lane center
        left_line  = (350, 480, 390, 288)
        right_line = (580, 480, 540, 288)
        assert self.detector._check_departure(left_line, right_line, width) is True

    def test_departure_no_lines(self):
        """No alert when either line is missing."""
        assert self.detector._check_departure(None, None, 640) is False
        assert self.detector._check_departure((100, 480, 150, 288), None, 640) is False

    def test_ema_smoothing_converges(self):
        """Smoothed line converges toward the new value over iterations."""
        self.detector.left_fit_avg = (100, 480, 120, 288)
        result = self.detector._smooth_line((200, 480, 220, 288),
                                            self.detector.left_fit_avg)
        # Should be between old and new — not equal to either extreme
        assert 100 < result[0] < 200
        assert 120 < result[2] < 220

    def test_ema_smoothing_no_history(self):
        """Without history, smoothing returns the new line unchanged."""
        new_line = (150, 480, 170, 288)
        result = self.detector._smooth_line(new_line, None)
        assert result == new_line

    def test_average_lines_stores_ema(self):
        """After processing a valid line, the corresponding avg is stored."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # slope = (480-300)/(200-100) = +1.8 → right lane
        lines = np.array([[[100, 300, 200, 480]]])
        self.detector._average_lines(frame, lines)
        assert self.detector.right_fit_avg is not None


# ─── Drowsiness Detection Tests ───────────────────────────────

class TestDrowsinessDetector:
    def setup_method(self):
        # Patch MediaPipe to avoid requiring a real camera/GPU
        with patch("mediapipe.solutions.face_mesh") as mock_fm:
            mock_fm.FaceMesh.return_value = MagicMock()
            mock_fm.FaceMesh.return_value.process.return_value = \
                MagicMock(multi_face_landmarks=None)
            from modules.drowsiness_detection import DrowsinessDetector
            self.detector = DrowsinessDetector()

    def test_ear_formula(self):
        """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)."""
        # Construct a simple eye: horizontal width=10, both vertical distances=2
        landmarks = {
            0: (0, 0),    # p1 — left corner
            1: (3, -2),   # p2 — top-left
            2: (7, -2),   # p3 — top-right
            3: (10, 0),   # p4 — right corner
            4: (7, 2),    # p5 — bottom-right
            5: (3, 2),    # p6 — bottom-left
        }
        eye_indices = [0, 1, 2, 3, 4, 5]
        ear = self.detector._calculate_ear(landmarks, eye_indices)
        # vertical_1 = dist(p2, p6) = dist((3,-2),(3,2)) = 4
        # vertical_2 = dist(p3, p5) = dist((7,-2),(7,2)) = 4
        # horizontal = dist(p1, p4) = dist((0,0),(10,0)) = 10
        # EAR = (4 + 4) / (2 * 10) = 0.4
        assert abs(ear - 0.4) < 1e-6

    def test_mar_zero_width_safety(self):
        """MAR returns 0.0 when mouth width is effectively zero (float-safe)."""
        landmarks = {
            0: (5, 0),    # left
            1: (5, 0),    # right — same point, width ≈ 0
            4: (5, -3),   # top
            5: (5, 3),    # bottom
        }
        mouth_indices = [0, 1, 0, 0, 4, 5, 0, 0]
        mar = self.detector._calculate_mar(landmarks, mouth_indices)
        assert mar == 0.0

    def test_mar_formula(self):
        """MAR = mouth_height / mouth_width."""
        landmarks = {
            0: (0, 0),    # left
            1: (10, 0),   # right  → width = 10
            4: (5, -3),   # top
            5: (5, 3),    # bottom → height = 6
        }
        mouth_indices = [0, 1, 0, 0, 4, 5, 0, 0]
        mar = self.detector._calculate_mar(landmarks, mouth_indices)
        assert abs(mar - 0.6) < 1e-6

    def test_drowsiness_frame_counter_increments(self):
        """Counter increments when EAR stays below threshold."""
        from config import EAR_THRESHOLD
        low_ear = EAR_THRESHOLD - 0.05
        self.detector._check_drowsiness(low_ear)
        assert self.detector.drowsy_frame_count == 1

    def test_drowsiness_frame_counter_resets(self):
        """Counter resets when EAR rises above threshold."""
        from config import EAR_THRESHOLD
        self.detector.drowsy_frame_count = 10
        self.detector._check_drowsiness(EAR_THRESHOLD + 0.05)
        assert self.detector.drowsy_frame_count == 0

    def test_drowsiness_alert_triggers_after_threshold(self):
        """Alert fires only after DROWSY_FRAME_COUNT consecutive frames."""
        from config import EAR_THRESHOLD, DROWSY_FRAME_COUNT
        low_ear = EAR_THRESHOLD - 0.05
        for _ in range(DROWSY_FRAME_COUNT - 1):
            result = self.detector._check_drowsiness(low_ear)
            assert result is False
        result = self.detector._check_drowsiness(low_ear)
        assert result is True

    def test_yawn_alert_triggers_after_threshold(self):
        """Yawn alert fires only after YAWN_FRAME_COUNT consecutive frames."""
        from config import MAR_THRESHOLD, YAWN_FRAME_COUNT
        high_mar = MAR_THRESHOLD + 0.1
        for _ in range(YAWN_FRAME_COUNT - 1):
            result = self.detector._check_yawning(high_mar)
            assert result is False
        result = self.detector._check_yawning(high_mar)
        assert result is True


# ─── Collision Warning Tests ───────────────────────────────────

class TestCollisionWarner:
    def setup_method(self):
        from modules.collision_warning import CollisionWarner
        self.warner = CollisionWarner()

    def test_box_area(self):
        """Box area = width × height."""
        area = self.warner._box_area((10, 10, 110, 60))
        assert area == 100 * 50

    def test_safe_when_no_vehicles(self):
        """Returns SAFE immediately when no vehicles detected."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, level = self.warner.process(frame, [])
        assert level == "SAFE"

    def test_safe_level_small_box(self):
        """Small bounding box (<15% frame) resolves to SAFE."""
        # 10% of 640x480 = ~30720 px² → use a 100x100 box (10240/307200 = 3.3%)
        vehicle = {"class_id": 2, "label": "Car", "confidence": 0.9,
                   "bbox": (0, 0, 100, 100)}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            _, level = self.warner.process(frame, [vehicle])
        assert level == "SAFE"

    def test_warning_level_after_frames(self):
        """Vehicle between 15-35% triggers WARNING after 5 frames."""
        from config import COLLISION_BOX_THRESHOLD
        # 25% of 307200 = 76800 → sqrt ≈ 277x277 box
        vehicle = {"class_id": 2, "label": "Car", "confidence": 0.9,
                   "bbox": (0, 0, 277, 277)}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            _, level = self.warner.process(frame, [vehicle])
        assert level == "WARNING"

    def test_danger_level_after_frames(self):
        """Vehicle >35% triggers DANGER after 5 frames."""
        # 40% of 307200 = 122880 → sqrt ≈ 350x350 box
        vehicle = {"class_id": 2, "label": "Car", "confidence": 0.9,
                   "bbox": (0, 0, 350, 350)}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            _, level = self.warner.process(frame, [vehicle])
        assert level == "DANGER"

    def test_state_holds_during_buildup(self):
        """While count < 5, danger_level stays at previous value (no SAFE flash)."""
        from modules.collision_warning import SAFE
        vehicle = {"class_id": 2, "label": "Car", "confidence": 0.9,
                   "bbox": (0, 0, 350, 350)}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # First frame: count=1, should return previous state (SAFE), not flash
        _, level = self.warner.process(frame, [vehicle])
        assert level == SAFE   # previous state was SAFE — correct hold behaviour

    def test_closest_vehicle_is_largest(self):
        """_find_closest returns the vehicle with the largest bounding box."""
        vehicles = [
            {"class_id": 2, "bbox": (0, 0, 50, 50)},
            {"class_id": 2, "bbox": (0, 0, 200, 200)},
            {"class_id": 2, "bbox": (0, 0, 100, 100)},
        ]
        frame_area = 480 * 640
        closest = self.warner._find_closest(vehicles, frame_area)
        assert closest["bbox"] == (0, 0, 200, 200)
