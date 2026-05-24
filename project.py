import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Traffic Helmet Detection",
    page_icon="🪖",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Exo+2:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

.stApp {
    background: #020412;
    min-height: 100vh;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(0,242,254,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(79,172,254,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(120,80,255,0.05) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

.main-title {
    text-align: center;
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00f2fe, #4facfe, #a78bfa, #00f2fe);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    margin: 0;
    padding-top: 1.5rem;
    letter-spacing: 3px;
    text-transform: uppercase;
}

@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 300% center; }
}

.subtitle {
    text-align: center;
    color: #4a5568;
    font-size: 0.9rem;
    margin-top: 0.5rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 300;
}

.divider {
    text-align: center;
    color: #4facfe;
    font-size: 1.2rem;
    margin: 0.5rem 0 2rem 0;
    opacity: 0.6;
    letter-spacing: 8px;
}

.cyber-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(79,172,254,0.05) 100%);
    border: 1px solid rgba(79,172,254,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}

.cyber-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, #4facfe, #00f2fe, transparent);
    animation: scan-line 3s linear infinite;
}

@keyframes scan-line {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    color: #4facfe;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 1.5rem;
}

.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 0.5rem;
    text-align: center;
}

.stat-card.total   { border-bottom: 2px solid #63b3ed; }
.stat-card.helmet  { border-bottom: 2px solid #48bb78; }
.stat-card.danger  { border-bottom: 2px solid #fc8181; }
.stat-card.rider   { border-bottom: 2px solid #f6ad55; }

.stat-number { font-family: 'Orbitron', monospace; font-size: 2.2rem; font-weight: 900; margin: 0; line-height: 1; }
.stat-label  { font-size: 0.7rem; color: #718096; margin: 6px 0 0; letter-spacing: 2px; text-transform: uppercase; }

.total-num  { color: #63b3ed; }
.helmet-num { color: #48bb78; }
.danger-num { color: #fc8181; }
.rider-num  { color: #f6ad55; }

.verdict-safe {
    background: linear-gradient(135deg, rgba(72,187,120,0.15), rgba(72,187,120,0.05));
    border: 1px solid rgba(72,187,120,0.4);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #68d391;
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-align: center;
    margin-top: 1rem;
}

.verdict-danger {
    background: linear-gradient(135deg, rgba(252,129,129,0.15), rgba(252,129,129,0.05));
    border: 1px solid rgba(252,129,129,0.4);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #fc8181;
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-align: center;
    margin-top: 1rem;
}

.empty-state { text-align: center; padding: 4rem 2rem; color: #2d3748; }
.empty-icon  { font-size: 4rem; display: block; margin-bottom: 1rem; animation: float 3s ease-in-out infinite; }

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-12px); }
}

.empty-title {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    color: #4a5568;
    letter-spacing: 2px;
}

div[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid rgba(79,172,254,0.2);
    width: 100%;
}

.stButton > button {
    background: linear-gradient(90deg, #00f2fe, #4facfe, #a78bfa) !important;
    background-size: 200% auto !important;
    color: #020412 !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.4s ease !important;
}

[data-testid="stSidebar"] {
    background: rgba(2,4,18,0.95) !important;
    border-right: 1px solid rgba(79,172,254,0.15) !important;
}

.sidebar-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #4facfe;
    letter-spacing: 2px;
    text-align: center;
    padding: 1rem 0 0.5rem;
}

.sidebar-section {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    color: #4facfe;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem;
    opacity: 0.8;
}

.legend-item { display: flex; align-items: center; gap: 10px; font-size: 0.82rem; color: #718096; margin: 6px 0; }
.legend-dot  { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

div[data-testid="stFileUploader"] {
    background: rgba(79,172,254,0.03) !important;
    border: 1px solid rgba(79,172,254,0.15) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "Bike-Helmet-Detectionv2/weights/best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "❌ Model weights not found at `Bike-Helmet-Detectionv2/weights/best.pt`.\n\n"
            "Run this once before starting the app:\n"
            "```\ngit clone https://github.com/Viddesh1/Bike-Helmet-Detectionv2.git\n```"
        )
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

COLORS = {
    "helmet":     (34,  197,  94),
    "head":       (239,  68,  68),
    "no helmet":  (239,  68,  68),
    "bike rider": (251, 146,  60),
    "rider":      (251, 146,  60),
}

def run_detection(pil_image: Image.Image, conf_thresh: float):
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    results  = model.predict(img_bgr, conf=conf_thresh, verbose=False)
    counts: dict[str, int] = {}

    for box in results[0].boxes:
        cls_id = int(box.cls)
        label  = model.names[cls_id].lower()
        counts[label] = counts.get(label, 0) + 1
        conf   = float(box.conf)
        color  = COLORS.get(label, (255, 255, 0))
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
        text = f"{label.upper()}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img_bgr, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
        cv2.putText(img_bgr, text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), counts


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚙ SETTINGS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Confidence</div>', unsafe_allow_html=True)
    conf_thresh = st.selectbox(
        "Confidence threshold",
        [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        index=3,
        label_visibility="collapsed"
    )
    st.markdown('<div class="sidebar-section">Legend</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="legend-item"><div class="legend-dot" style="background:#48bb78"></div>Helmet detected</div>
    <div class="legend-item"><div class="legend-dot" style="background:#fc8181"></div>No helmet (head)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f6ad55"></div>Bike rider</div>
    """, unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🪖 Helmet Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">YOLOv8 · Real-Time · Traffic Monitoring</p>', unsafe_allow_html=True)
st.markdown('<div class="divider">◆ ◆ ◆</div>', unsafe_allow_html=True)


# ── Main layout ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📡 Input Feed</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
    else:
        st.markdown("""
        <div style="border:2px dashed rgba(79,172,254,0.25);border-radius:16px;padding:3rem 2rem;text-align:center;">
            <span style="font-size:3rem;display:block;margin-bottom:0.75rem;">📤</span>
            <p style="font-family:'Orbitron',monospace;font-size:0.85rem;color:#4a5568;letter-spacing:2px;">DROP IMAGE HERE</p>
            <p style="color:#2d3748;font-size:0.75rem;margin-top:0.5rem;">JPG · PNG · WEBP</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    detect_btn = st.button("⬡  SCAN FOR HELMETS", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


with col2:
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🎯 Detection Output</div>', unsafe_allow_html=True)

    if detect_btn and image:
        with st.spinner("Scanning..."):
            result_img, counts = run_detection(image, conf_thresh)

        st.image(result_img, use_container_width=True)

        helmets   = counts.get("helmet", 0)
        no_helmet = counts.get("head", 0) + counts.get("no helmet", 0)
        riders    = counts.get("bike rider", 0) + counts.get("rider", 0)
        total     = sum(counts.values())

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card total">
                <p class="stat-number total-num">{total}</p>
                <p class="stat-label">Total</p>
            </div>
            <div class="stat-card helmet">
                <p class="stat-number helmet-num">{helmets}</p>
                <p class="stat-label">Helmet</p>
            </div>
            <div class="stat-card danger">
                <p class="stat-number danger-num">{no_helmet}</p>
                <p class="stat-label">No Helmet</p>
            </div>
            <div class="stat-card rider">
                <p class="stat-number rider-num">{riders}</p>
                <p class="stat-label">Riders</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if no_helmet > 0:
            st.markdown(
                f'<div class="verdict-danger">⚠ VIOLATION — {no_helmet} RIDER(S) WITHOUT HELMET</div>',
                unsafe_allow_html=True
            )
        elif helmets > 0:
            st.markdown(
                '<div class="verdict-safe">✔ COMPLIANT — ALL RIDERS WEARING HELMETS</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="verdict-danger">◌ NO DETECTIONS — TRY A CLEARER IMAGE</div>',
                unsafe_allow_html=True
            )

    elif detect_btn and not image:
        st.warning("⚠️ Please upload an image first.")
    else:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">🎯</span>
            <p class="empty-title">AWAITING SCAN</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
