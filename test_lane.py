# test_drowsiness.py
import cv2
from modules.drowsiness_detection import DrowsinessDetector

detector = DrowsinessDetector()

cap = cv2.VideoCapture(0)
print("✅ Drowsiness Detection running...")
print("💡 Try closing your eyes slowly to trigger alert")
print("💡 Try yawning to trigger yawn alert")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result_frame, drowsy, yawn = detector.process(frame)

    if drowsy:
        print("🚨 DROWSINESS ALERT!")
    if yawn:
        print("😮 YAWNING DETECTED!")

    cv2.imshow("Drowsiness Detection Test", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()