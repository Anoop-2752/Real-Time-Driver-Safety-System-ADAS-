# streamlit_app.py

import cv2
import streamlit as st
import streamlit.components.v1 as components
import time
from collections import deque
from modules.lane_detection import LaneDetector
from modules.object_detection import ObjectDetector
from modules.drowsiness_detection import DrowsinessDetector
from modules.collision_warning import CollisionWarner
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, VIDEO_FRONT, VIDEO_DRIVER

st.set_page_config(
    page_title="Driver Safety System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* remove default streamlit top padding */
    .block-container { padding-top: 0.5rem !important; padding-bottom: 0rem !important; }

    /* hide hamburger and footer */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    .alert-danger {
        background: #3d0000;
        border: 2px solid #ff4444;
        border-radius: 6px;
        padding: 6px 10px;
        color: #ff4444;
        font-weight: bold;
        text-align: center;
        font-size: 13px;
    }
    .alert-safe {
        background: #003d00;
        border: 2px solid #00cc44;
        border-radius: 6px;
        padding: 6px 10px;
        color: #00cc44;
        font-weight: bold;
        text-align: center;
        font-size: 13px;
    }
    div[data-testid="stMetricValue"] { font-size: 20px !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; }
    div[data-testid="stMetricDelta"] { font-size: 11px !important; }
    div[data-testid="column"] { padding: 0 4px !important; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "running":      False,
        "alert_log":    deque(maxlen=50),
        "fps":          0.0,
        "ear":          0.0,
        "mar":          0.0,
        "lane_alert":   False,
        "drowsy_alert": False,
        "yawn_alert":   False,
        "danger_level": "SAFE",
        "counts":       {},
        "total_alerts": 0,
        "frame_count":  0,
        "start_time":   time.time(),
        "was_alerted":  False,
        "cap":          None,
        "cap_driver":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


@st.cache_resource
def load_modules():
    return {
        "lane":      LaneDetector(),
        "object":    ObjectDetector(),
        "drowsy":    DrowsinessDetector(),
        "collision": CollisionWarner()
    }


def render_sidebar():
    st.sidebar.title("🚗 Driver Safety System")
    st.sidebar.markdown("---")

    use_demo     = st.sidebar.toggle("Demo Mode (test videos)", value=True)
    camera_index = st.sidebar.selectbox("Camera", [0, 1, 2], index=0, disabled=use_demo)
    show_lanes     = st.sidebar.toggle("Lane Detection",       value=True)
    show_objects   = st.sidebar.toggle("Object Detection",     value=True)
    show_drowsy    = st.sidebar.toggle("Drowsiness Detection", value=True)
    show_collision = st.sidebar.toggle("Collision Warning",    value=True)

    st.sidebar.markdown("---")
    elapsed = int(time.time() - st.session_state.start_time)
    st.sidebar.metric("Runtime",      f"{elapsed // 60:02d}:{elapsed % 60:02d}")
    st.sidebar.metric("Total Alerts", st.session_state.total_alerts)
    st.sidebar.metric("Frames",       st.session_state.frame_count)
    st.sidebar.markdown("---")
    st.sidebar.caption("Built by Anoop Krishna · YOLOv8 · MediaPipe · OpenCV")

    return camera_index, use_demo, show_lanes, show_objects, show_drowsy, show_collision


def status_card(label, status, is_alert):
    css = "alert-danger" if is_alert else "alert-safe"
    icon = "⚠" if is_alert else "✓"
    st.markdown(
        f"<div class='{css}'>{icon} {label}<br><small>{status}</small></div>",
        unsafe_allow_html=True
    )


def process_frame(front_frame, driver_frame, modules, show_lanes, show_objects, show_drowsy, show_collision):
    lane_alert   = False
    drowsy_alert = False
    yawn_alert   = False
    danger_level = "SAFE"
    counts       = {}
    detections   = []

    front_frame  = cv2.resize(front_frame,  (640, 360))
    driver_frame = cv2.resize(driver_frame, (640, 360))

    processed = front_frame.copy()

    if show_lanes:
        processed, lane_alert = modules["lane"].process(processed)
    if show_objects:
        processed, detections, counts = modules["object"].process(processed)
    if show_collision and detections:
        processed, danger_level = modules["collision"].process(processed, detections)

    drv = driver_frame.copy()
    if show_drowsy:
        drv, drowsy_alert, yawn_alert = modules["drowsy"].process(drv)

    return processed, drv, lane_alert, drowsy_alert, yawn_alert, danger_level, counts


def main():
    camera_index, use_demo, show_lanes, show_objects, show_drowsy, show_collision = render_sidebar()

    # compact header + controls on same row
    h_col, b1, b2, b3 = st.columns([3, 1, 1, 1])
    with h_col:
        st.markdown("<h3 style='margin:0; padding:4px 0; color:#ffffff;'>🚗 Real-Time Driver Safety System — ADAS</h3>", unsafe_allow_html=True)
    with b1:
        if st.button("▶ Start", type="primary", use_container_width=True):
            st.session_state.running      = True
            st.session_state.start_time   = time.time()
            st.session_state.frame_count  = 0
            st.session_state.total_alerts = 0
    with b2:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.running = False
            for key in ["cap", "cap_driver"]:
                if st.session_state[key] is not None:
                    st.session_state[key].release()
                    st.session_state[key] = None
    with b3:
        if st.button("🗑 Clear Log", use_container_width=True):
            st.session_state.alert_log.clear()

    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

    # main layout: front video | driver video | right panel
    vid1, vid2, panel = st.columns([2, 1, 1])

    with vid1:
        st.caption("📷 Front Camera — Road View")
        front_placeholder = st.empty()

    with vid2:
        st.caption("👤 Driver Monitor")
        driver_placeholder = st.empty()

    with panel:
        panel_placeholder = st.empty()

    # video loop
    if st.session_state.running:
        modules = load_modules()

        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            if use_demo:
                cap        = cv2.VideoCapture(VIDEO_FRONT)
                cap_driver = cv2.VideoCapture(VIDEO_DRIVER)
            else:
                cap = cv2.VideoCapture(camera_index)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap_driver = cap

            if not cap.isOpened():
                st.error("❌ Camera / video file not found!")
                st.session_state.running = False
                return
            st.session_state.cap        = cap
            st.session_state.cap_driver = cap_driver
        else:
            cap        = st.session_state.cap
            cap_driver = st.session_state.cap_driver

        frame_time  = time.time()
        frame_count = 0

        while st.session_state.running:
            ret_f, front_frame = cap.read()
            if not ret_f:
                if use_demo:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_f, front_frame = cap.read()
                if not ret_f:
                    st.error("❌ Front video read failed!")
                    break

            ret_d, driver_frame = cap_driver.read()
            if not ret_d:
                if use_demo:
                    cap_driver.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_d, driver_frame = cap_driver.read()
                if not ret_d:
                    st.error("❌ Driver video read failed!")
                    break

            frame_count += 1
            if frame_count % 2 == 0:
                continue

            (front_frame, driver_frame,
             lane_alert, drowsy_alert,
             yawn_alert, danger_level, counts) = process_frame(
                front_frame, driver_frame, modules,
                show_lanes, show_objects,
                show_drowsy, show_collision
            )

            now = time.time()
            fps = 1.0 / (now - frame_time + 1e-9)
            frame_time = now

            st.session_state.fps          = fps
            st.session_state.lane_alert   = lane_alert
            st.session_state.drowsy_alert = drowsy_alert
            st.session_state.yawn_alert   = yawn_alert
            st.session_state.danger_level = danger_level
            st.session_state.counts       = counts
            st.session_state.frame_count += 1
            st.session_state.ear          = modules["drowsy"].ear_value
            st.session_state.mar          = modules["drowsy"].mar_value

            is_alerted = lane_alert or drowsy_alert or yawn_alert or danger_level != "SAFE"
            if is_alerted and not st.session_state.was_alerted:
                st.session_state.total_alerts += 1
                alerts = []
                if lane_alert:             alerts.append("Lane Departure")
                if drowsy_alert:           alerts.append("Drowsiness")
                if yawn_alert:             alerts.append("Yawning")
                if danger_level != "SAFE": alerts.append(f"Collision {danger_level}")
                st.session_state.alert_log.append({
                    "time": time.strftime("%H:%M:%S"),
                    "msg":  " | ".join(alerts)
                })

                if danger_level == "DANGER":
                    freq, duration = 1200, 0.3
                elif drowsy_alert:
                    freq, duration = 440, 1.0
                elif lane_alert:
                    freq, duration = 880, 0.5
                elif yawn_alert:
                    freq, duration = 660, 0.6
                else:
                    freq, duration = 800, 0.4

                components.html(f"""
                <script>
                (function() {{
                    var ctx = new (window.AudioContext || window.webkitAudioContext)();
                    var osc = ctx.createOscillator();
                    var gain = ctx.createGain();
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.frequency.value = {freq};
                    osc.type = 'sine';
                    gain.gain.setValueAtTime(0.3, ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + {duration});
                    osc.start(ctx.currentTime);
                    osc.stop(ctx.currentTime + {duration});
                }})();
                </script>
                """, height=0)
            st.session_state.was_alerted = is_alerted

            front_placeholder.image(
                cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )
            driver_placeholder.image(
                cv2.cvtColor(driver_frame, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

            dl  = st.session_state.danger_level
            ear = st.session_state.ear
            mar = st.session_state.mar

            lane_css  = "alert-danger" if lane_alert   else "alert-safe"
            drv_css   = "alert-danger" if drowsy_alert else "alert-safe"
            yawn_css  = "alert-danger" if yawn_alert   else "alert-safe"
            coll_css  = "alert-danger" if dl != "SAFE" else "alert-safe"
            ear_delta = f"<span style='color:#ff4444;font-size:11px'>▼ LOW</span>"  if ear < 0.25 and ear > 0 else ""
            mar_delta = f"<span style='color:#ff4444;font-size:11px'>▲ HIGH</span>" if mar > 0.6            else ""

            log_html = ""
            for entry in list(st.session_state.alert_log)[::-1][:8]:
                icon = "🚨" if "DANGER" in entry["msg"] or "Drowsiness" in entry["msg"] else "⚠️"
                log_html += f"<div style='font-size:11px;margin:2px 0'><code>{entry['time']}</code> {icon} {entry['msg']}</div>"
            if not log_html:
                log_html = "<div style='font-size:12px;color:#888'>No alerts yet</div>"

            panel_placeholder.markdown(f"""
<div style='font-size:13px'>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:6px'>
    <div class='{lane_css}'>{"⚠" if lane_alert   else "✓"} Lane<br><small>{"DEPARTURE!" if lane_alert   else "OK"}</small></div>
    <div class='{drv_css}' >{"⚠" if drowsy_alert else "✓"} Driver<br><small>{"DROWSY!"    if drowsy_alert else "OK"}</small></div>
    <div class='{yawn_css}'>{"⚠" if yawn_alert   else "✓"} Yawn<br><small>{"YES"         if yawn_alert   else "None"}</small></div>
    <div class='{coll_css}'>{"⚠" if dl != "SAFE" else "✓"} Collision<br><small>{dl}</small></div>
  </div>
  <hr style='margin:4px 0;border-color:#333'>
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;text-align:center;margin-bottom:6px'>
    <div><div style='color:#888;font-size:11px'>FPS</div><b>{fps:.1f}</b></div>
    <div><div style='color:#888;font-size:11px'>EAR</div><b>{ear:.2f}</b>{ear_delta}</div>
    <div><div style='color:#888;font-size:11px'>MAR</div><b>{mar:.2f}</b>{mar_delta}</div>
    <div><div style='color:#888;font-size:11px'>Objs</div><b>{sum(counts.values())}</b></div>
  </div>
  <hr style='margin:4px 0;border-color:#333'>
  <div style='color:#aaa;font-size:11px;margin-bottom:3px'>📋 Alert Log</div>
  {log_html}
</div>
""", unsafe_allow_html=True)

        st.info("⏹ System stopped")


if __name__ == "__main__":
    main()
