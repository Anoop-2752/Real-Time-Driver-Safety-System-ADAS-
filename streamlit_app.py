# streamlit_app.py

import cv2
import numpy as np
import streamlit as st
import time
import threading
from collections import deque
from modules.lane_detection import LaneDetector
from modules.object_detection import ObjectDetector
from modules.drowsiness_detection import DrowsinessDetector
from modules.collision_warning import CollisionWarner
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Safety System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2d3250;
    }

    .alert-danger {
        background: #3d0000;
        border: 2px solid #ff4444;
        border-radius: 8px;
        padding: 10px;
        color: #ff4444;
        font-weight: bold;
        text-align: center;
    }

    .alert-warning {
        background: #3d2000;
        border: 2px solid #ff8c00;
        border-radius: 8px;
        padding: 10px;
        color: #ff8c00;
        font-weight: bold;
        text-align: center;
    }

    .alert-safe {
        background: #003d00;
        border: 2px solid #00cc44;
        border-radius: 8px;
        padding: 10px;
        color: #00cc44;
        font-weight: bold;
        text-align: center;
    }

    .status-header {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ──────────────────────────────────
def init_session_state():
    defaults = {
        "running":        False,
        "alert_log":      deque(maxlen=50),
        "fps":            0.0,
        "ear":            0.0,
        "mar":            0.0,
        "lane_alert":     False,
        "drowsy_alert":   False,
        "yawn_alert":     False,
        "danger_level":   "SAFE",
        "counts":         {},
        "total_alerts":   0,
        "frame_count":    0,
        "start_time":     time.time(),
        "was_alerted":    False,   # tracks previous frame's alert state for per-event counting
        "cap":            None,    # VideoCapture stored to prevent leaks on rerun
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ─── Load Modules (cached) ─────────────────────────────────────
@st.cache_resource
def load_modules():
    return {
        "lane":      LaneDetector(),
        "object":    ObjectDetector(),
        "drowsy":    DrowsinessDetector(),
        "collision": CollisionWarner()
    }


# ─── Sidebar ───────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/car.png",
        width=80
    )
    st.sidebar.title("🚗 Driver Safety System")
    st.sidebar.markdown("---")

    st.sidebar.subheader("⚙️ Settings")

    camera_index = st.sidebar.selectbox(
        "Camera Source",
        [0, 1, 2],
        index=0,
        help="Select camera index"
    )

    show_lanes = st.sidebar.toggle("Lane Detection", value=True)
    show_objects = st.sidebar.toggle("Object Detection", value=True)
    show_drowsy = st.sidebar.toggle("Drowsiness Detection", value=True)
    show_collision = st.sidebar.toggle("Collision Warning", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Session Stats")

    elapsed = int(time.time() - st.session_state.start_time)
    mins = elapsed // 60
    secs = elapsed % 60

    st.sidebar.metric("Runtime", f"{mins:02d}:{secs:02d}")
    st.sidebar.metric("Total Alerts", st.session_state.total_alerts)
    st.sidebar.metric("Frames", st.session_state.frame_count)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Built by:** Anoop Krishna\n\n"
        "**Stack:** YOLOv8 · MediaPipe · OpenCV\n\n"
        "**Purpose:** ADAS Portfolio Project"
    )

    return camera_index, show_lanes, show_objects, show_drowsy, show_collision


# ─── Status Card ───────────────────────────────────────────────
def status_card(label, status, is_alert):
    if is_alert:
        html = f"""
        <div class='alert-danger'>
            ⚠️ {label}<br><small>{status}</small>
        </div>"""
    else:
        html = f"""
        <div class='alert-safe'>
            ✅ {label}<br><small>{status}</small>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ─── Alert Log ─────────────────────────────────────────────────
def render_alert_log():
    st.subheader("📋 Alert Log")

    if not st.session_state.alert_log:
        st.info("No alerts yet — system running normally")
        return

    log_list = list(st.session_state.alert_log)[::-1]

    for entry in log_list[:10]:
        icon = "🚨" if "DANGER" in entry or "DROWSY" in entry else "⚠️"
        st.markdown(f"`{entry['time']}`  {icon}  **{entry['msg']}**")


# ─── Process Single Frame ──────────────────────────────────────
def process_frame(frame, modules,
                   show_lanes, show_objects,
                   show_drowsy, show_collision):

    lane_alert    = False
    drowsy_alert  = False
    yawn_alert    = False
    danger_level  = "SAFE"
    counts        = {}
    detections    = []

    processed = frame.copy()

    if show_lanes:
        processed, lane_alert = modules["lane"].process(processed)

    if show_objects:
        processed, detections, counts = modules["object"].process(processed)

    if show_collision and detections:
        processed, danger_level = modules["collision"].process(
            processed, detections
        )

    driver_frame = frame.copy()
    if show_drowsy:
        driver_frame, drowsy_alert, yawn_alert = \
            modules["drowsy"].process(driver_frame)

    return (processed, driver_frame,
            lane_alert, drowsy_alert,
            yawn_alert, danger_level, counts)


# ─── Main App ──────────────────────────────────────────────────
def main():
    # Header
    st.markdown(
        "<h1 style='text-align:center; color:#ffffff;'>"
        "🚗 Real-Time Driver Safety System</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:#aaaaaa;'>"
        "ADAS · Computer Vision · Deep Learning</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Sidebar
    camera_index, show_lanes, show_objects, \
        show_drowsy, show_collision = render_sidebar()

    # Controls
    col_start, col_stop, col_clear = st.columns([1, 1, 2])

    with col_start:
        if st.button("▶ Start System", type="primary",
                     use_container_width=True):
            st.session_state.running = True
            st.session_state.start_time = time.time()
            st.session_state.frame_count = 0
            st.session_state.total_alerts = 0

    with col_stop:
        if st.button("⏹ Stop System", use_container_width=True):
            st.session_state.running = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None

    with col_clear:
        if st.button("🗑 Clear Alert Log", use_container_width=True):
            st.session_state.alert_log.clear()

    st.markdown("---")

    # Status cards row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        status_card("Lane Detection",
                    "DEPARTURE!" if st.session_state.lane_alert else "Normal",
                    st.session_state.lane_alert)
    with c2:
        status_card("Driver Status",
                    "DROWSY!" if st.session_state.drowsy_alert else "Alert",
                    st.session_state.drowsy_alert)
    with c3:
        is_yawn = st.session_state.yawn_alert
        status_card("Yawn Detection",
                    "YAWNING!" if is_yawn else "None",
                    is_yawn)
    with c4:
        dl = st.session_state.danger_level
        status_card("Collision Risk",
                    dl,
                    dl != "SAFE")

    st.markdown("---")

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("FPS", f"{st.session_state.fps:.1f}")
    with m2:
        ear_val = st.session_state.ear
        st.metric("EAR",
                  f"{ear_val:.3f}",
                  delta="⚠ LOW" if ear_val < 0.25 and ear_val > 0 else None,
                  delta_color="inverse")
    with m3:
        mar_val = st.session_state.mar
        st.metric("MAR",
                  f"{mar_val:.3f}",
                  delta="⚠ HIGH" if mar_val > 0.6 else None,
                  delta_color="inverse")
    with m4:
        obj_count = sum(st.session_state.counts.values())
        st.metric("Objects Detected", obj_count)

    st.markdown("---")

    # Video feeds
    vid_col, log_col = st.columns([2, 1])

    with vid_col:
        st.subheader("📷 Live Feed")
        front_placeholder  = st.empty()
        driver_placeholder = st.empty()

    with log_col:
        render_alert_log()

    # ── Main Loop ──────────────────────────────────────────────
    if st.session_state.running:
        modules = load_modules()

        # Reuse existing capture if already open, otherwise create a new one
        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            if not cap.isOpened():
                st.error("❌ Camera not found!")
                st.session_state.running = False
                return
            st.session_state.cap = cap
        else:
            cap = st.session_state.cap

        frame_time = time.time()

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Camera read failed!")
                break

            # Process frame
            (front_frame, driver_frame,
             lane_alert, drowsy_alert,
             yawn_alert, danger_level, counts) = process_frame(
                frame, modules,
                show_lanes, show_objects,
                show_drowsy, show_collision
            )

            # FPS
            now = time.time()
            fps = 1.0 / (now - frame_time + 1e-9)
            frame_time = now

            # Update session state
            st.session_state.fps           = fps
            st.session_state.lane_alert    = lane_alert
            st.session_state.drowsy_alert  = drowsy_alert
            st.session_state.yawn_alert    = yawn_alert
            st.session_state.danger_level  = danger_level
            st.session_state.counts        = counts
            st.session_state.frame_count  += 1
            st.session_state.ear           = modules["drowsy"].ear_value
            st.session_state.mar           = modules["drowsy"].mar_value

            # Alert counter: only increment on transition from no-alert → alert
            is_alerted = (lane_alert or drowsy_alert or yawn_alert
                          or danger_level != "SAFE")
            if is_alerted and not st.session_state.was_alerted:
                st.session_state.total_alerts += 1
                alerts = []
                if lane_alert:              alerts.append("Lane Departure")
                if drowsy_alert:            alerts.append("Drowsiness")
                if yawn_alert:              alerts.append("Yawning")
                if danger_level != "SAFE":  alerts.append(f"Collision {danger_level}")
                st.session_state.alert_log.append({
                    "time": time.strftime("%H:%M:%S"),
                    "msg":  " | ".join(alerts)
                })
            st.session_state.was_alerted = is_alerted

            # Convert to RGB for streamlit
            front_rgb  = cv2.cvtColor(front_frame,  cv2.COLOR_BGR2RGB)
            driver_rgb = cv2.cvtColor(driver_frame, cv2.COLOR_BGR2RGB)

            # Display frames
            front_placeholder.image(
                front_rgb,
                caption="Front Camera — Road View",
                use_column_width=True
            )
            driver_placeholder.image(
                driver_rgb,
                caption="Driver Monitor",
                use_column_width=True
            )

        st.info("⏹ System stopped")


if __name__ == "__main__":
    main()