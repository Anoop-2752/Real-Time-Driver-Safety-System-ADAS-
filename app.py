# app.py

import cv2
import numpy as np
import pygame
import time
import threading
from modules.lane_detection import LaneDetector
from modules.object_detection import ObjectDetector
from modules.drowsiness_detection import DrowsinessDetector
from modules.collision_warning import CollisionWarner
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    SOUND_LANE_ALERT, SOUND_DROWSY_ALERT, SOUND_COLLISION_ALERT,
    COLOR_WHITE, COLOR_RED, COLOR_GREEN, COLOR_YELLOW, COLOR_ORANGE
)

# ─── Dashboard Dimensions ──────────────────────────────────────
DASH_W = 1280
DASH_H = 600
FRONT_W = 800
FRONT_H = 480
DRIVER_W = 460
DRIVER_H = 240


class DriverSafetySystem:
    def __init__(self):
        print("🚗 Initializing Driver Safety System...")

        # Initialize all modules
        self.lane_detector = LaneDetector()
        self.object_detector = ObjectDetector()
        self.drowsiness_detector = DrowsinessDetector()
        self.collision_warner = CollisionWarner()

        # Initialize pygame audio
        pygame.mixer.init()
        self.sounds = self._load_sounds()

        # Alert cooldown tracking
        self.last_lane_alert = 0
        self.last_drowsy_alert = 0
        self.last_collision_alert = 0
        self.alert_cooldown = 3.0

        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()

        print("✅ All modules loaded successfully!")
        print("─" * 50)

    # ─── Load Sounds ───────────────────────────────────────────
    def _load_sounds(self):
        sounds = {}
        try:
            sounds["lane"] = pygame.mixer.Sound(SOUND_LANE_ALERT)
            sounds["drowsy"] = pygame.mixer.Sound(SOUND_DROWSY_ALERT)
            sounds["collision"] = pygame.mixer.Sound(SOUND_COLLISION_ALERT)
            print("✅ Sound files loaded!")
        except Exception as e:
            print(f"⚠️  Sound loading failed: {e}")
            print("   Continuing without audio...")
        return sounds

    # ─── Play Alert Sound ──────────────────────────────────────
    def _play_alert(self, alert_type):
        now = time.time()
        cooldown_map = {
            "lane":      "last_lane_alert",
            "drowsy":    "last_drowsy_alert",
            "collision": "last_collision_alert"
        }
        attr = cooldown_map.get(alert_type)
        if attr and (now - getattr(self, attr)) > self.alert_cooldown:
            setattr(self, attr, now)
            if alert_type in self.sounds:
                threading.Thread(
                    target=self.sounds[alert_type].play,
                    daemon=True
                ).start()

    # ─── Process Front Camera ──────────────────────────────────
    def _process_front(self, frame):
        lane_frame, lane_alert = self.lane_detector.process(frame)
        obj_frame, detections, counts = self.object_detector.process(lane_frame)
        final_frame, danger_level = self.collision_warner.process(
            obj_frame, detections
        )
        return final_frame, lane_alert, detections, counts, danger_level

    # ─── Process Driver Camera ─────────────────────────────────
    def _process_driver(self, frame):
        result_frame, drowsy_alert, yawn_alert = \
            self.drowsiness_detector.process(frame)
        return result_frame, drowsy_alert, yawn_alert

    # ─── Calculate FPS ─────────────────────────────────────────
    def _calculate_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0

    # ─── Handle Audio Alerts ───────────────────────────────────
    def _handle_alerts(self, lane_alert, drowsy_alert, danger_level):
        if lane_alert:
            self._play_alert("lane")
        if drowsy_alert:
            self._play_alert("drowsy")
        if danger_level == "DANGER":
            self._play_alert("collision")

    # ─── Build Dashboard ───────────────────────────────────────
    def _build_dashboard(self, front_frame, driver_frame,
                          lane_alert, drowsy_alert,
                          yawn_alert, danger_level, counts):

        # Resize camera frames
        front_resized  = cv2.resize(front_frame,  (FRONT_W, FRONT_H))
        driver_resized = cv2.resize(driver_frame, (DRIVER_W, DRIVER_H))

        # Black canvas
        dashboard = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)

        # ── Header Bar ─────────────────────────────────────────
        cv2.rectangle(dashboard, (0, 0), (DASH_W, 55), (30, 30, 30), -1)
        cv2.putText(dashboard,
                    "Real-Time Driver Safety System  |  ADAS",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WHITE, 2)

        fps = self._calculate_fps()
        cv2.putText(dashboard, f"FPS: {fps:.1f}",
                    (DASH_W - 120, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)

        # ── Front Camera ───────────────────────────────────────
        dashboard[60:60 + FRONT_H, 0:FRONT_W] = front_resized

        # ── Driver Camera ──────────────────────────────────────
        drv_x = FRONT_W + 10
        drv_y = 60
        dashboard[drv_y:drv_y + DRIVER_H,
                  drv_x:drv_x + DRIVER_W] = driver_resized

        # Driver cam label
        cv2.putText(dashboard, "Driver Monitor",
                    (drv_x + 5, drv_y + DRIVER_H + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

        # ── Status Panel ───────────────────────────────────────
        panel_x = FRONT_W + 10
        panel_y = drv_y + DRIVER_H + 35
        panel_w = DRIVER_W
        panel_h = 220

        cv2.rectangle(dashboard,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (40, 40, 40), -1)
        cv2.rectangle(dashboard,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (100, 100, 100), 1)

        # Panel title
        cv2.putText(dashboard, "SYSTEM STATUS",
                    (panel_x + 10, panel_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLOR_YELLOW, 2)

        # Divider line
        cv2.line(dashboard,
                 (panel_x + 5, panel_y + 38),
                 (panel_x + panel_w - 5, panel_y + 38),
                 (100, 100, 100), 1)

        # Status rows
        statuses = [
            {
                "label": "Lane",
                "text":  "DEPARTURE !" if lane_alert else "OK",
                "color": COLOR_RED if lane_alert else COLOR_GREEN,
                "icon":  "⚠" if lane_alert else "✓"
            },
            {
                "label": "Driver",
                "text":  "DROWSY !" if drowsy_alert else "Alert",
                "color": COLOR_RED if drowsy_alert else COLOR_GREEN,
                "icon":  "⚠" if drowsy_alert else "✓"
            },
            {
                "label": "Yawn",
                "text":  "DETECTED" if yawn_alert else "None",
                "color": COLOR_ORANGE if yawn_alert else COLOR_GREEN,
                "icon":  "⚠" if yawn_alert else "✓"
            },
            {
                "label": "Collision",
                "text":  danger_level,
                "color": (COLOR_RED    if danger_level == "DANGER"
                          else COLOR_ORANGE if danger_level == "WARNING"
                          else COLOR_GREEN),
                "icon":  ("🚨" if danger_level == "DANGER"
                          else "⚠" if danger_level == "WARNING"
                          else "✓")
            }
        ]

        row_y = panel_y + 60
        for s in statuses:
            # Label
            cv2.putText(dashboard,
                        f"{s['label']}:",
                        (panel_x + 10, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
            # Value
            cv2.putText(dashboard,
                        f"{s['icon']} {s['text']}",
                        (panel_x + 120, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, s["color"], 2)
            row_y += 35

        # Object counts
        if counts:
            count_str = "  ".join([f"{k}:{v}" for k, v in counts.items()])
            cv2.putText(dashboard,
                        f"Objects: {count_str}",
                        (panel_x + 10, panel_y + panel_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_WHITE, 1)

        # ── Bottom Status Bar ──────────────────────────────────
        bar_y = DASH_H - 50
        any_alert = (lane_alert or drowsy_alert or
                     yawn_alert or danger_level != "SAFE")

        bar_color = (0, 0, 120) if any_alert else (0, 80, 0)
        cv2.rectangle(dashboard,
                      (0, bar_y),
                      (DASH_W, DASH_H),
                      bar_color, -1)

        if any_alert:
            msg = "⚠  ALERT ACTIVE — Please take immediate action!"
            msg_color = (0, 200, 255)
        else:
            msg = "✓  All systems normal — Drive safe!"
            msg_color = COLOR_GREEN

        cv2.putText(dashboard, msg,
                    (20, bar_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, msg_color, 2)

        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(dashboard, timestamp,
                    (DASH_W - 120, bar_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 1)

        return dashboard

    # ─── Main Run Loop ─────────────────────────────────────────
    def run(self):
        print("\n🚀 Starting Driver Safety System...")
        print("📷 Camera initializing...")

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            print("❌ Camera not found! Check CAMERA_INDEX in config.py")
            return

        print("✅ Camera ready!")
        print("Press Q to quit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame capture failed!")
                break

            # Process front view
            front_frame, lane_alert, detections, counts, danger_level = \
                self._process_front(frame)

            # Process driver view (same cam for testing)
            driver_frame, drowsy_alert, yawn_alert = \
                self._process_driver(frame)

            # Audio alerts
            self._handle_alerts(lane_alert, drowsy_alert, danger_level)

            # Build and show dashboard
            dashboard = self._build_dashboard(
                front_frame, driver_frame,
                lane_alert, drowsy_alert,
                yawn_alert, danger_level, counts
            )

            cv2.imshow("Driver Safety System", dashboard)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n👋 Shutting down...")
                break

        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("✅ System shutdown complete")


# ─── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    system = DriverSafetySystem()
    system.run()