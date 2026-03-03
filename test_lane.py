# test_lane.py
import cv2
from modules.lane_detection import LaneDetector

detector = LaneDetector()

# Test on webcam (press Q to quit)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result_frame, alert = detector.process(frame)

    cv2.imshow("Lane Detection Test", result_frame)

    if alert:
        print("⚠️ LANE DEPARTURE DETECTED!")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()