# test_object.py
import cv2
from modules.object_detection import ObjectDetector

detector = ObjectDetector()

# Test on webcam (press Q to quit)
cap = cv2.VideoCapture("assets/test_videos/dashcam.mp4")

print("✅ Object Detection running... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result_frame, detections, counts = detector.process(frame)

    # Print detections in terminal
    if detections:
        print(f"Detected: {counts}")

    cv2.imshow("Object Detection Test", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()