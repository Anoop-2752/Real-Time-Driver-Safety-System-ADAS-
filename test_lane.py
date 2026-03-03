# test_collision.py
import cv2
from modules.object_detection import ObjectDetector
from modules.collision_warning import CollisionWarner

detector = ObjectDetector()
warner = CollisionWarner()

cap = cv2.VideoCapture("assets/test_videos/dashcam.mp4")
print("✅ Collision Warning running...")
print("💡 Move close to camera to trigger warning")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # First get detections
    obj_frame, detections, counts = detector.process(frame)

    # Then run collision warning
    result_frame, danger_level = warner.process(obj_frame, detections)

    if danger_level != "SAFE":
        print(f"🚨 Danger Level: {danger_level}")

    cv2.imshow("Collision Warning Test", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()