# modules/lane_detection.py

import cv2
import numpy as np
from config import (
    LANE_COLOR, LANE_THICKNESS, ROI_HEIGHT_RATIO,
    COLOR_RED, COLOR_GREEN, COLOR_WHITE, SMOOTHING_ALPHA
)


class LaneDetector:
    def __init__(self):
        self.left_fit_avg = None
        self.right_fit_avg = None
        self.departure_alert = False

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        height, width = frame.shape[:2]
        edges = self._preprocess(frame)
        masked = self._region_of_interest(edges, height, width)
        lines = self._detect_lines(masked)
        left_line, right_line = self._average_lines(frame, lines)
        lane_frame = self._draw_lanes(frame, left_line, right_line, height, width)
        self.departure_alert = self._check_departure(left_line, right_line, width)
        lane_frame = self._draw_status(lane_frame, self.departure_alert)
        return lane_frame, self.departure_alert

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def _region_of_interest(self, edges, height, width):
        mask = np.zeros_like(edges)
        roi_vertices = np.array([[
            (0, height),
            (width * 0.45, height * ROI_HEIGHT_RATIO),
            (width * 0.55, height * ROI_HEIGHT_RATIO),
            (width, height)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        return cv2.bitwise_and(edges, mask)

    def _detect_lines(self, masked):
        return cv2.HoughLinesP(
            masked,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=40,
            maxLineGap=150
        )

    def _average_lines(self, frame, lines):
        left_lines = []
        right_lines = []

        if lines is None:
            return None, None

        height, width = frame.shape[:2]

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            if slope < -0.3:
                left_lines.append((slope, intercept))
            elif slope > 0.3:
                right_lines.append((slope, intercept))

        left_line = self._make_line(frame, left_lines)
        right_line = self._make_line(frame, right_lines)

        # EMA smoothing — reduces frame-to-frame jitter
        if left_line is not None:
            left_line = self._smooth_line(left_line, self.left_fit_avg)
            self.left_fit_avg = left_line

        if right_line is not None:
            right_line = self._smooth_line(right_line, self.right_fit_avg)
            self.right_fit_avg = right_line

        # Fall back to last known good line if none detected this frame
        if left_line is None:
            left_line = self.left_fit_avg
        if right_line is None:
            right_line = self.right_fit_avg

        return left_line, right_line

    def _make_line(self, frame, lines):
        if not lines:
            return None

        height = frame.shape[0]
        slope, intercept = np.mean(lines, axis=0)

        y1 = height
        y2 = int(height * ROI_HEIGHT_RATIO)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return (x1, y1, x2, y2)

    def _smooth_line(self, new_line: tuple, prev_avg) -> tuple:
        if prev_avg is None:
            return new_line
        return tuple(
            int(SMOOTHING_ALPHA * n + (1 - SMOOTHING_ALPHA) * p)
            for n, p in zip(new_line, prev_avg)
        )

    def _draw_lanes(self, frame, left_line, right_line, height, width):
        lane_overlay = np.zeros_like(frame)

        if left_line is not None:
            cv2.line(lane_overlay, (left_line[0], left_line[1]),
                     (left_line[2], left_line[3]), LANE_COLOR, LANE_THICKNESS)

        if right_line is not None:
            cv2.line(lane_overlay, (right_line[0], right_line[1]),
                     (right_line[2], right_line[3]), LANE_COLOR, LANE_THICKNESS)

        # Fill lane corridor with semi-transparent green
        if left_line is not None and right_line is not None:
            pts = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ], dtype=np.int32)
            cv2.fillPoly(lane_overlay, [pts], (0, 100, 0))

        return cv2.addWeighted(frame, 1, lane_overlay, 0.4, 0)

    def _check_departure(self, left_line, right_line, width):
        if left_line is None or right_line is None:
            return False

        lane_center = (left_line[0] + right_line[0]) // 2
        frame_center = width // 2
        deviation = abs(lane_center - frame_center)

        # Alert if vehicle drifts more than 15% of frame width off-center
        return deviation > width * 0.15

    def _draw_status(self, frame, alert):
        text  = "⚠ LANE DEPARTURE WARNING!" if alert else "✓ Lane: OK"
        color = COLOR_RED if alert else COLOR_GREEN
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame
