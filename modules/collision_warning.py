# modules/collision_warning.py

import cv2
import numpy as np
from config import (
    COLLISION_BOX_THRESHOLD,
    COLLISION_COLOR_SAFE,
    COLLISION_COLOR_WARNING,
    COLLISION_COLOR_DANGER,
    COLOR_WHITE, COLOR_RED, COLOR_GREEN, COLOR_YELLOW
)


SAFE    = "SAFE"
WARNING = "WARNING"
DANGER  = "DANGER"

# Vehicle classes only — pedestrians excluded from collision logic
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorbike, bus, truck


class CollisionWarner:
    def __init__(self):
        self.danger_level = SAFE
        self.closest_vehicle = None
        self.warning_frame_count = 0
        self.danger_frame_count = 0

    def process(self, frame: np.ndarray, detections: list) -> tuple[np.ndarray, str]:
        h, w = frame.shape[:2]
        frame_area = h * w

        vehicles = [d for d in detections if d["class_id"] in VEHICLE_CLASSES]
        self.closest_vehicle = self._find_closest(vehicles, frame_area)
        self.danger_level = self._calculate_danger(self.closest_vehicle, frame_area)

        frame = self._draw_collision_overlay(frame, self.closest_vehicle)
        frame = self._draw_status(frame, w, h)
        return frame, self.danger_level

    def _find_closest(self, vehicles, frame_area):
        if not vehicles:
            return None
        # Largest bounding box = closest vehicle (perspective proxy)
        return max(vehicles, key=lambda d: self._box_area(d["bbox"]))

    def _box_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _calculate_danger(self, vehicle, frame_area):
        if vehicle is None:
            self.warning_frame_count = 0
            self.danger_frame_count = 0
            return SAFE

        ratio = self._box_area(vehicle["bbox"]) / frame_area

        if ratio > COLLISION_BOX_THRESHOLD:
            self.danger_frame_count += 1
            self.warning_frame_count = 0
            if self.danger_frame_count >= 5:
                return DANGER

        elif ratio > 0.15:
            self.warning_frame_count += 1
            self.danger_frame_count = 0
            if self.warning_frame_count >= 5:
                return WARNING

        else:
            self.warning_frame_count = 0
            self.danger_frame_count = 0
            return SAFE

        # Hold previous state until frame threshold is met — prevents flickering
        return self.danger_level

    def _draw_collision_overlay(self, frame, vehicle):
        if vehicle is None:
            return frame

        x1, y1, x2, y2 = vehicle["bbox"]
        ratio = self._box_area(vehicle["bbox"]) / (frame.shape[0] * frame.shape[1])

        if ratio > COLLISION_BOX_THRESHOLD:
            color, thickness = COLLISION_COLOR_DANGER, 4
        elif ratio > 0.15:
            color, thickness = COLLISION_COLOR_WARNING, 3
        else:
            color, thickness = COLLISION_COLOR_SAFE, 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Corner bracket markers
        corner_len = 20
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

        cv2.putText(frame, f"Proximity: {ratio * 100:.1f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def _draw_status(self, frame, w, h):
        if self.danger_level == DANGER:
            cv2.rectangle(frame, (w - 310, 0), (w, 70), (0, 0, 180), -1)
            cv2.putText(frame, "🚨 COLLISION", (w - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)
            cv2.putText(frame, "   WARNING!", (w - 300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

        elif self.danger_level == WARNING:
            cv2.rectangle(frame, (w - 310, 0), (w, 70), (0, 100, 200), -1)
            cv2.putText(frame, "⚠ VEHICLE", (w - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)
            cv2.putText(frame, "  CLOSE!", (w - 300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

        else:
            cv2.rectangle(frame, (w - 210, 0), (w, 40), (0, 100, 0), -1)
            cv2.putText(frame, "✓ Collision: SAFE",
                        (w - 200, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2)

        return frame
