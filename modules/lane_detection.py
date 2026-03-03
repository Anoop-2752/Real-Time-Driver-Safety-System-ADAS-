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

    # ─── Main Process Function ─────────────────────────────────
    def process(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Main function — takes raw frame, returns annotated frame + alert status
        """
        height, width = frame.shape[:2]

        # Step 1: Preprocess
        edges = self._preprocess(frame)

        # Step 2: Mask region of interest
        masked = self._region_of_interest(edges, height, width)

        # Step 3: Detect lines
        lines = self._detect_lines(masked)

        # Step 4: Average lines into left and right lane
        left_line, right_line = self._average_lines(frame, lines)

        # Step 5: Draw lanes on frame
        lane_frame = self._draw_lanes(frame, left_line, right_line, height, width)

        # Step 6: Check for lane departure
        self.departure_alert = self._check_departure(left_line, right_line, width)

        # Step 7: Draw alert status on frame
        lane_frame = self._draw_status(lane_frame, self.departure_alert)

        return lane_frame, self.departure_alert

    # ─── Step 1: Preprocessing ────────────────────────────────
    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    # ─── Step 2: Region of Interest ───────────────────────────
    def _region_of_interest(self, edges, height, width):
        mask = np.zeros_like(edges)

        # Trapezoid shape focusing on bottom half of frame (road area)
        roi_vertices = np.array([[
            (0, height),
            (width * 0.45, height * ROI_HEIGHT_RATIO),
            (width * 0.55, height * ROI_HEIGHT_RATIO),
            (width, height)
        ]], dtype=np.int32)

        cv2.fillPoly(mask, roi_vertices, 255)
        masked = cv2.bitwise_and(edges, mask)
        return masked

    # ─── Step 3: Hough Line Detection ─────────────────────────
    def _detect_lines(self, masked):
        lines = cv2.HoughLinesP(
            masked,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=40,
            maxLineGap=150
        )
        return lines

    # ─── Step 4: Average & Extrapolate Lines ──────────────────
    def _average_lines(self, frame, lines):
        left_lines = []
        right_lines = []

        if lines is None:
            return None, None

        height, width = frame.shape[:2]

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                continue  # skip vertical lines

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            if slope < -0.3:          # Negative slope = left lane
                left_lines.append((slope, intercept))
            elif slope > 0.3:         # Positive slope = right lane
                right_lines.append((slope, intercept))

        left_line = self._make_line(frame, left_lines)
        right_line = self._make_line(frame, right_lines)

        # Apply EMA smoothing — keep previous value if no new line detected
        if left_line is not None:
            left_line = self._smooth_line(left_line, self.left_fit_avg)
            self.left_fit_avg = left_line

        if right_line is not None:
            right_line = self._smooth_line(right_line, self.right_fit_avg)
            self.right_fit_avg = right_line

        # Fall back to last known good line if current frame has no detection
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
        """Apply EMA smoothing to reduce frame-to-frame jitter."""
        if prev_avg is None:
            return new_line
        return tuple(
            int(SMOOTHING_ALPHA * n + (1 - SMOOTHING_ALPHA) * p)
            for n, p in zip(new_line, prev_avg)
        )

    # ─── Step 5: Draw Lanes ───────────────────────────────────
    def _draw_lanes(self, frame, left_line, right_line, height, width):
        lane_overlay = np.zeros_like(frame)

        if left_line is not None:
            cv2.line(lane_overlay, (left_line[0], left_line[1]),
                     (left_line[2], left_line[3]), LANE_COLOR, LANE_THICKNESS)

        if right_line is not None:
            cv2.line(lane_overlay, (right_line[0], right_line[1]),
                     (right_line[2], right_line[3]), LANE_COLOR, LANE_THICKNESS)

        # Fill lane area in green (semi-transparent)
        if left_line is not None and right_line is not None:
            pts = np.array([
                [left_line[0], left_line[1]],
                [left_line[2], left_line[3]],
                [right_line[2], right_line[3]],
                [right_line[0], right_line[1]]
            ], dtype=np.int32)
            cv2.fillPoly(lane_overlay, [pts], (0, 100, 0))

        result = cv2.addWeighted(frame, 1, lane_overlay, 0.4, 0)
        return result

    # ─── Step 6: Lane Departure Check ─────────────────────────
    def _check_departure(self, left_line, right_line, width):
        if left_line is None or right_line is None:
            return False

        # Calculate lane center vs frame center
        lane_center = (left_line[0] + right_line[0]) // 2
        frame_center = width // 2
        deviation = abs(lane_center - frame_center)

        # If deviation is more than 15% of frame width → departure
        if deviation > width * 0.15:
            return True
        return False

    # ─── Step 7: Draw Status ──────────────────────────────────
    def _draw_status(self, frame, alert):
        if alert:
            text = "⚠ LANE DEPARTURE WARNING!"
            color = COLOR_RED
        else:
            text = "✓ Lane: OK"
            color = COLOR_GREEN

        cv2.putText(frame, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame