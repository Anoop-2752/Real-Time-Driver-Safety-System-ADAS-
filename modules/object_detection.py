# modules/object_detection.py

import cv2
from ultralytics import YOLO
from config import (
    YOLO_MODEL_PATH, YOLO_CONFIDENCE, DETECTION_CLASSES,
    COLOR_RED, COLOR_GREEN, COLOR_YELLOW, COLOR_WHITE, COLOR_ORANGE
)


class ObjectDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)

        self.target_labels = {
            0: "Person",
            1: "Bicycle",
            2: "Car",
            3: "Motorbike",
            5: "Bus",
            7: "Truck"
        }

        self.class_colors = {
            0: (0, 0, 255),    # Person
            1: (0, 255, 255),  # Bicycle
            2: (0, 255, 0),    # Car
            3: (255, 165, 0),  # Motorbike
            5: (255, 0, 0),    # Bus
            7: (128, 0, 128)   # Truck
        }

        self.detection_counts = {}

    def process(self, frame):
        results = self.model(frame, conf=YOLO_CONFIDENCE, verbose=False)
        detections = self._parse_detections(results)
        annotated_frame = self._draw_detections(frame.copy(), detections)
        self.detection_counts = self._count_objects(detections)
        annotated_frame = self._draw_summary(annotated_frame, self.detection_counts)
        return annotated_frame, detections, self.detection_counts

    def _parse_detections(self, results):
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                if class_id not in DETECTION_CLASSES:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.target_labels.get(class_id, "Unknown")

                detections.append({
                    "class_id":   class_id,
                    "label":      label,
                    "confidence": confidence,
                    "bbox":       (x1, y1, x2, y2)
                })

        return detections

    def _draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = self.class_colors.get(det["class_id"], COLOR_WHITE)
            label_text = f"{det['label']} {det['confidence']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

        return frame

    def _count_objects(self, detections):
        counts = {}
        for det in detections:
            label = det["label"]
            counts[label] = counts.get(label, 0) + 1
        return counts

    def _draw_summary(self, frame, counts):
        if not counts:
            return frame

        panel_height = 30 + (len(counts) * 25)
        cv2.rectangle(frame, (10, 55), (200, 55 + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 55), (200, 55 + panel_height), COLOR_WHITE, 1)
        cv2.putText(frame, "Detections:", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

        y_pos = 100
        for label, count in counts.items():
            cv2.putText(frame, f"  {label}: {count}", (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
            y_pos += 25

        return frame
