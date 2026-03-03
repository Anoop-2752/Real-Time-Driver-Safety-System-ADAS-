# modules/drowsiness_detection.py

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
from config import (
    EAR_THRESHOLD, MAR_THRESHOLD, DROWSY_FRAME_COUNT,
    COLOR_RED, COLOR_GREEN, COLOR_YELLOW, COLOR_WHITE, COLOR_ORANGE
)


class DrowsinessDetector:
    def __init__(self):
        # MediaPipe Face Mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Eye landmark indices (MediaPipe 468 landmarks)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        # Mouth landmark indices
        self.MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

        # State tracking
        self.drowsy_frame_count = 0
        self.yawn_frame_count = 0
        self.drowsy_alert = False
        self.yawn_alert = False
        self.ear_value = 0.0
        self.mar_value = 0.0

    # ─── Main Process Function ─────────────────────────────────
    def process(self, frame):
        """
        Main function — takes driver camera frame, returns annotated frame + alerts
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        self.drowsy_alert = False
        self.yawn_alert = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]

                # Get landmark coordinates
                landmarks = self._get_landmarks(face_landmarks, w, h)

                # Calculate EAR
                left_ear = self._calculate_ear(landmarks, self.LEFT_EYE)
                right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE)
                self.ear_value = (left_ear + right_ear) / 2.0

                # Calculate MAR
                self.mar_value = self._calculate_mar(landmarks, self.MOUTH)

                # Draw eye and mouth contours
                frame = self._draw_contours(frame, landmarks)

                # Check drowsiness
                self.drowsy_alert = self._check_drowsiness(self.ear_value)

                # Check yawning
                self.yawn_alert = self._check_yawning(self.mar_value)

                # Draw metrics on frame
                frame = self._draw_metrics(frame)

        else:
            # No face detected
            frame = self._draw_no_face(frame)

        # Draw alert status
        frame = self._draw_status(frame)

        return frame, self.drowsy_alert, self.yawn_alert

    # ─── Get Landmark Coordinates ──────────────────────────────
    def _get_landmarks(self, face_landmarks, w, h):
        landmarks = {}
        for idx, lm in enumerate(face_landmarks.landmark):
            landmarks[idx] = (int(lm.x * w), int(lm.y * h))
        return landmarks

    # ─── Calculate EAR ────────────────────────────────────────
    def _calculate_ear(self, landmarks, eye_indices):
        p1 = landmarks[eye_indices[0]]
        p2 = landmarks[eye_indices[1]]
        p3 = landmarks[eye_indices[2]]
        p4 = landmarks[eye_indices[3]]
        p5 = landmarks[eye_indices[4]]
        p6 = landmarks[eye_indices[5]]

        vertical_1 = distance.euclidean(p2, p6)
        vertical_2 = distance.euclidean(p3, p5)
        horizontal = distance.euclidean(p1, p4)

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    # ─── Calculate MAR ────────────────────────────────────────
    def _calculate_mar(self, landmarks, mouth_indices):
        top = landmarks[mouth_indices[4]]
        bottom = landmarks[mouth_indices[5]]
        left = landmarks[mouth_indices[0]]
        right = landmarks[mouth_indices[1]]

        mouth_height = distance.euclidean(top, bottom)
        mouth_width = distance.euclidean(left, right)

        if mouth_width == 0:
            return 0.0

        mar = mouth_height / mouth_width
        return mar

    # ─── Draw Eye & Mouth Contours ─────────────────────────────
    def _draw_contours(self, frame, landmarks):
        # Draw left eye
        left_eye_pts = np.array(
            [landmarks[i] for i in self.LEFT_EYE], dtype=np.int32
        )
        cv2.polylines(frame, [left_eye_pts], True, COLOR_GREEN, 1)

        # Draw right eye
        right_eye_pts = np.array(
            [landmarks[i] for i in self.RIGHT_EYE], dtype=np.int32
        )
        cv2.polylines(frame, [right_eye_pts], True, COLOR_GREEN, 1)

        # Draw mouth
        mouth_pts = np.array(
            [landmarks[i] for i in self.MOUTH], dtype=np.int32
        )
        cv2.polylines(frame, [mouth_pts], True, COLOR_YELLOW, 1)

        return frame

    # ─── Check Drowsiness ──────────────────────────────────────
    def _check_drowsiness(self, ear):
        if ear < EAR_THRESHOLD:
            self.drowsy_frame_count += 1
            if self.drowsy_frame_count >= DROWSY_FRAME_COUNT:
                return True
        else:
            self.drowsy_frame_count = 0
        return False

    # ─── Check Yawning ─────────────────────────────────────────
    def _check_yawning(self, mar):
        if mar > MAR_THRESHOLD:
            self.yawn_frame_count += 1
            if self.yawn_frame_count >= 10:
                return True
        else:
            self.yawn_frame_count = 0
        return False

    # ─── Draw Metrics Panel ────────────────────────────────────
    def _draw_metrics(self, frame):
        # Background panel
        cv2.rectangle(frame, (10, 55), (220, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 55), (220, 130), COLOR_WHITE, 1)

        # EAR value
        ear_color = COLOR_RED if self.ear_value < EAR_THRESHOLD else COLOR_GREEN
        cv2.putText(frame, f"EAR: {self.ear_value:.3f}",
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_color, 2)

        # MAR value
        mar_color = COLOR_ORANGE if self.mar_value > MAR_THRESHOLD else COLOR_GREEN
        cv2.putText(frame, f"MAR: {self.mar_value:.3f}",
                    (15, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mar_color, 2)

        return frame

    # ─── Draw Alert Status ─────────────────────────────────────
    def _draw_status(self, frame):
        h, w = frame.shape[:2]

        if self.drowsy_alert:
            # Big red alert banner
            cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 180), -1)
            cv2.putText(frame, "⚠ DROWSINESS ALERT! WAKE UP!",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, COLOR_WHITE, 2)

        elif self.yawn_alert:
            cv2.rectangle(frame, (0, h - 60), (w, h), (0, 140, 255), -1)
            cv2.putText(frame, "⚠ YAWNING DETECTED! Take a break!",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, COLOR_WHITE, 2)

        else:
            cv2.putText(frame, "✓ Driver: Alert",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, COLOR_GREEN, 2)

        return frame

    # ─── No Face Detected ──────────────────────────────────────
    def _draw_no_face(self, frame):
        cv2.putText(frame, "⚠ No Face Detected!",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, COLOR_ORANGE, 2)
        return frame