# modules/drowsiness_detection.py

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
from config import (
    EAR_THRESHOLD, MAR_THRESHOLD, DROWSY_FRAME_COUNT, YAWN_FRAME_COUNT,
    SMOOTHING_ALPHA, COLOR_RED, COLOR_GREEN, COLOR_YELLOW, COLOR_WHITE, COLOR_ORANGE
)


class DrowsinessDetector:
    def __init__(self):
        # mediapipe 0.10.x removed mp.solutions — try legacy first, fall back to Tasks API
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._legacy = True
        except AttributeError:
            self._init_tasks_api()
            self._legacy = False

        self.LEFT_EYE  = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]

        self.drowsy_frame_count = 0
        self.yawn_frame_count   = 0
        self.drowsy_alert = False
        self.yawn_alert   = False
        self.ear_value = 0.0
        self.mar_value = 0.0

    def _init_tasks_api(self):
        import os
        import urllib.request
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = "models/face_landmarker.task"
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame):
        self.drowsy_alert = False
        self.yawn_alert   = False

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._legacy:
            results = self.face_mesh.process(rgb)
            face_list = results.multi_face_landmarks or []
            landmarks_list = [
                {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(fl.landmark)}
                for fl in face_list
            ]
        else:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self.face_landmarker.detect(mp_img)
            landmarks_list = [
                {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face)}
                for face in (results.face_landmarks or [])
            ]

        if landmarks_list:
            landmarks = landmarks_list[0]

            raw_ear = (self._calculate_ear(landmarks, self.LEFT_EYE) +
                       self._calculate_ear(landmarks, self.RIGHT_EYE)) / 2.0
            self.ear_value = SMOOTHING_ALPHA * raw_ear + (1 - SMOOTHING_ALPHA) * self.ear_value

            raw_mar = self._calculate_mar(landmarks, self.MOUTH)
            self.mar_value = SMOOTHING_ALPHA * raw_mar + (1 - SMOOTHING_ALPHA) * self.mar_value

            frame = self._draw_contours(frame, landmarks)
            self.drowsy_alert = self._check_drowsiness(self.ear_value)
            self.yawn_alert   = self._check_yawning(self.mar_value)
            frame = self._draw_metrics(frame)
        else:
            cv2.putText(frame, "No Face Detected",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ORANGE, 2)

        return self._draw_status(frame), self.drowsy_alert, self.yawn_alert

    def _calculate_ear(self, landmarks, eye_indices):
        p1, p2, p3 = landmarks[eye_indices[0]], landmarks[eye_indices[1]], landmarks[eye_indices[2]]
        p4, p5, p6 = landmarks[eye_indices[3]], landmarks[eye_indices[4]], landmarks[eye_indices[5]]
        return (distance.euclidean(p2, p6) + distance.euclidean(p3, p5)) / (2.0 * distance.euclidean(p1, p4))

    def _calculate_mar(self, landmarks, mouth_indices):
        top    = landmarks[mouth_indices[4]]
        bottom = landmarks[mouth_indices[5]]
        left   = landmarks[mouth_indices[0]]
        right  = landmarks[mouth_indices[1]]
        mouth_width = distance.euclidean(left, right)
        if abs(mouth_width) < 1e-6:
            return 0.0
        return distance.euclidean(top, bottom) / mouth_width

    def _draw_contours(self, frame, landmarks):
        cv2.polylines(frame, [np.array([landmarks[i] for i in self.LEFT_EYE],  dtype=np.int32)], True, COLOR_GREEN,  1)
        cv2.polylines(frame, [np.array([landmarks[i] for i in self.RIGHT_EYE], dtype=np.int32)], True, COLOR_GREEN,  1)
        cv2.polylines(frame, [np.array([landmarks[i] for i in self.MOUTH],     dtype=np.int32)], True, COLOR_YELLOW, 1)
        return frame

    def _check_drowsiness(self, ear):
        if ear < EAR_THRESHOLD:
            self.drowsy_frame_count += 1
            if self.drowsy_frame_count >= DROWSY_FRAME_COUNT:
                return True
        else:
            self.drowsy_frame_count = 0
        return False

    def _check_yawning(self, mar):
        if mar > MAR_THRESHOLD:
            self.yawn_frame_count += 1
            if self.yawn_frame_count >= YAWN_FRAME_COUNT:
                return True
        else:
            self.yawn_frame_count = 0
        return False

    def _draw_metrics(self, frame):
        cv2.rectangle(frame, (10, 55), (220, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 55), (220, 130), COLOR_WHITE, 1)
        ear_color = COLOR_RED if self.ear_value < EAR_THRESHOLD else COLOR_GREEN
        cv2.putText(frame, f"EAR: {self.ear_value:.3f}", (15, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_color,   2)
        mar_color = COLOR_ORANGE if self.mar_value > MAR_THRESHOLD else COLOR_GREEN
        cv2.putText(frame, f"MAR: {self.mar_value:.3f}", (15, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mar_color, 2)
        return frame

    def _draw_status(self, frame):
        h, w = frame.shape[:2]
        if self.drowsy_alert:
            cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 180), -1)
            cv2.putText(frame, "DROWSINESS ALERT! WAKE UP!",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)
        elif self.yawn_alert:
            cv2.rectangle(frame, (0, h - 60), (w, h), (0, 140, 255), -1)
            cv2.putText(frame, "YAWNING DETECTED! Take a break!",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
        else:
            cv2.putText(frame, "Driver: Alert",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2)
        return frame
