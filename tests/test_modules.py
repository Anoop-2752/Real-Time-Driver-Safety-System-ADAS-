# tests/test_modules.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestLaneDetector:
    def setup_method(self):
        from modules.lane_detection import LaneDetector
        self.detector = LaneDetector()

    def test_departure_centered_lanes(self):
        width = 640
        left_line  = (160, 480, 200, 288)
        right_line = (480, 480, 440, 288)
        assert self.detector._check_departure(left_line, right_line, width) is False

    def test_departure_off_center(self):
        width = 640
        left_line  = (350, 480, 390, 288)
        right_line = (580, 480, 540, 288)
        assert self.detector._check_departure(left_line, right_line, width) is True

    def test_departure_no_lines(self):
        assert self.detector._check_departure(None, None, 640) is False
        assert self.detector._check_departure((100, 480, 150, 288), None, 640) is False

    def test_ema_smoothing_converges(self):
        self.detector.left_fit_avg = (100, 480, 120, 288)
        result = self.detector._smooth_line((200, 480, 220, 288),
                                            self.detector.left_fit_avg)
        assert 100 < result[0] < 200
        assert 120 < result[2] < 220

    def test_ema_smoothing_no_history(self):
        new_line = (150, 480, 170, 288)
        result = self.detector._smooth_line(new_line, None)
        assert result == new_line

    def test_average_lines_stores_ema(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lines = np.array([[[100, 300, 200, 480]]])
        self.detector._average_lines(frame, lines)
        assert self.detector.right_fit_avg is not None


class TestDrowsinessDetector:
    def setup_method(self):
        with patch("mediapipe.solutions.face_mesh") as mock_fm:
            mock_fm.FaceMesh.return_value = MagicMock()
            mock_fm.FaceMesh.return_value.process.return_value = \
                MagicMock(multi_face_landmarks=None)
            from modules.drowsiness_detection import DrowsinessDetector
            self.detector = DrowsinessDetector()

    def test_ear_formula(self):
        # horizontal width=10, both vertical distances=4 → EAR = (4+4)/(2*10) = 0.4
        landmarks = {
            0: (0, 0),
            1: (3, -2),
            2: (7, -2),
            3: (10, 0),
            4: (7, 2),
            5: (3, 2),
        }
        ear = self.detector._calculate_ear(landmarks, [0, 1, 2, 3, 4, 5])
        assert abs(ear - 0.4) < 1e-6

    def test_mar_zero_width_safety(self):
        landmarks = {
            0: (5, 0),
            1: (5, 0),
            4: (5, -3),
            5: (5, 3),
        }
        mar = self.detector._calculate_mar(landmarks, [0, 1, 0, 0, 4, 5, 0, 0])
        assert mar == 0.0

    def test_mar_formula(self):
        # width=10, height=6 → MAR = 0.6
        landmarks = {
            0: (0, 0),
            1: (10, 0),
            4: (5, -3),
            5: (5, 3),
        }
        mar = self.detector._calculate_mar(landmarks, [0, 1, 0, 0, 4, 5, 0, 0])
        assert abs(mar - 0.6) < 1e-6

    def test_drowsiness_frame_counter_increments(self):
        from config import EAR_THRESHOLD
        self.detector._check_drowsiness(EAR_THRESHOLD - 0.05)
        assert self.detector.drowsy_frame_count == 1

    def test_drowsiness_frame_counter_resets(self):
        from config import EAR_THRESHOLD
        self.detector.drowsy_frame_count = 10
        self.detector._check_drowsiness(EAR_THRESHOLD + 0.05)
        assert self.detector.drowsy_frame_count == 0

    def test_drowsiness_alert_triggers_after_threshold(self):
        from config import EAR_THRESHOLD, DROWSY_FRAME_COUNT
        low_ear = EAR_THRESHOLD - 0.05
        for _ in range(DROWSY_FRAME_COUNT - 1):
            assert self.detector._check_drowsiness(low_ear) is False
        assert self.detector._check_drowsiness(low_ear) is True

    def test_yawn_alert_triggers_after_threshold(self):
        from config import MAR_THRESHOLD, YAWN_FRAME_COUNT
        high_mar = MAR_THRESHOLD + 0.1
        for _ in range(YAWN_FRAME_COUNT - 1):
            assert self.detector._check_yawning(high_mar) is False
        assert self.detector._check_yawning(high_mar) is True


class TestCollisionWarner:
    def setup_method(self):
        from modules.collision_warning import CollisionWarner
        self.warner = CollisionWarner()

    def test_box_area(self):
        assert self.warner._box_area((10, 10, 110, 60)) == 100 * 50

    def test_safe_when_no_vehicles(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, level = self.warner.process(frame, [])
        assert level == "SAFE"

    def test_safe_level_small_box(self):
        vehicle = {"class_id": 2, "label": "Car", "confidence": 0.9,
                   "bbox": (0, 0, 100, 100)}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            _, level = self.warner.process(frame, [vehicle])
        assert level == "SAFE"

    def test_warning_level_after_frames(self):
        vehicle = {"class_id": 2, "label": "Car", "confidence": 0.9,
                   "bbox": (0, 0, 277, 277)}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            _, level = self.warner.process(frame, [vehicle])
        assert level == "WARNING"

    def test_danger_level_after_frames(self):
        vehicle = {"class_id": 2, "label": "Car", "confidence": 0.9,
                   "bbox": (0, 0, 350, 350)}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            _, level = self.warner.process(frame, [vehicle])
        assert level == "DANGER"

    def test_state_holds_during_buildup(self):
        from modules.collision_warning import SAFE
        vehicle = {"class_id": 2, "label": "Car", "confidence": 0.9,
                   "bbox": (0, 0, 350, 350)}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, level = self.warner.process(frame, [vehicle])
        assert level == SAFE

    def test_closest_vehicle_is_largest(self):
        vehicles = [
            {"class_id": 2, "bbox": (0, 0, 50, 50)},
            {"class_id": 2, "bbox": (0, 0, 200, 200)},
            {"class_id": 2, "bbox": (0, 0, 100, 100)},
        ]
        closest = self.warner._find_closest(vehicles, 480 * 640)
        assert closest["bbox"] == (0, 0, 200, 200)
