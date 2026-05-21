import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras
import os
from PIL import Image
import gdown
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

@keras.saving.register_keras_serializable()
def preprocess(x):
    x = tf.cast(x, tf.float32)
    return preprocess_input(x)

MODEL_URL = "https://drive.google.com/uc?id=1p3veX7I7_6WBM97jOSfQpSGcxwIuijD1"
MODEL_LOCAL = "best_inceptionresnetv2_face_shape_fixed.keras"

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_LOCAL):
        with st.spinner("⬇️ Downloading model..."):
            gdown.download(MODEL_URL, MODEL_LOCAL, quiet=False)
    face_model = tf.keras.models.load_model(MODEL_LOCAL, custom_objects={'preprocess': preprocess})
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_model, face_cascade

classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

shape_info = {
    'Oval':   {'emoji': '🥚', 'color': '#4CAF50', 'desc': 'ใบหน้ารูปไข่ — สมดุลที่สุด เหมาะกับทุกทรงผม',
               'hair': 'ผมสั้นถึงกลาง เช่น blunt bob, shoulder-length, pixie cut, long layers และหน้าม้าปัดข้าง'},
    'Square': {'emoji': '⬛', 'color': '#2196F3', 'desc': 'ใบหน้าเหลี่ยม — กรามและหน้าผากกว้างพอกัน',
               'hair': 'ผมยาวปานกลางถึงยาว พร้อมไล่เลเยอร์หรือปลายฟุ้ง เช่น beach waves และหน้าม้านุ่มๆ'},
    'Round':  {'emoji': '⭕', 'color': '#FF9800', 'desc': 'ใบหน้ากลม — แก้มอิ่ม ใบหน้ากว้างและสั้น',
               'hair': 'ทรงเพิ่มความสูงให้ใบหน้า เช่น textured bob, long layers, แสกข้าง และ blunt bangs'},
    'Heart':  {'emoji': '❤️', 'color': '#E91E63', 'desc': 'ใบหน้ารูปหัวใจ — หน้าผากกว้าง คางแหลม',
               'hair': 'ผมยาวระดับไหล่ พร้อมเลเยอร์บริเวณกราม curtain bangs หรือ wispy bangs'},
    'Oblong': {'emoji': '📏', 'color': '#9C27B0', 'desc': 'ใบหน้ายาว — ยาวกว่ากว้างมาก',
               'hair': 'ลอนคลาย, loose curls, layered bob และหน้าม้าปัดข้างหรือ curtain bangs'},
}

# ─── Page config ───
st.set_page_config(page_title="Face Shape AI", page_icon="✨", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }

.main { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); min-height: 100vh; }

.hero {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #e94560, #f5a623, #00d2ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p { color: #a0aec0; font-size: 1rem; margin-top: 0; }

.upload-box {
    background: rgba(255,255,255,0.05);
    border: 2px dashed rgba(255,255,255,0.2);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    transition: all 0.3s;
}

.result-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.1);
}

.shape-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.5rem;
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.metric-box {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}
.metric-box .val { font-size: 1.6rem; font-weight: 700; color: #f5a623; }
.metric-box .lbl { font-size: 0.75rem; color: #a0aec0; margin-top: 2px; }

.hair-section {
    background: rgba(255,255,255,0.04);
    border-left: 4px solid #e94560;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.7;
}

.footer { text-align:center; color: #4a5568; font-size: 0.8rem; padding: 2rem 0 1rem; }

/* Override streamlit elements */
.stFileUploader > div { background: transparent !important; }
[data-testid="stFileUploadDropzone"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(255,255,255,0.2) !important;
    border-radius: 16px !important;
    color: #a0aec0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero ───
st.markdown("""
<div class="hero">
    <h1>✨ Face Shape AI</h1>
    <p>วิเคราะห์รูปใบหน้าและแนะนำทรงผมที่เหมาะกับคุณด้วย AI</p>
</div>
""", unsafe_allow_html=True)

face_shape_model, face_cascade = load_models()

# ─── Upload ───
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"],
                                  label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; color:#4a5568; padding: 1rem;">
        📸 อัปโหลดภาพใบหน้าเพื่อเริ่มวิเคราะห์
    </div>
    """, unsafe_allow_html=True)

os.makedirs("saved_results", exist_ok=True)

def predict_face_shape(img_pil):
    img = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_out = img.copy()

    img_resized = cv2.resize(img_bgr, (299, 299))
    img_input = np.expand_dims(img_resized, axis=0)
    pred = face_shape_model.predict(img_input, verbose=0)
    idx = np.argmax(pred)
    face_shape = classes[idx]
    confidence = pred[0][idx] * 100

    ratiog = 0.0
    score = 0.0
    face_detected = False

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        face_detected = True
        x, y, w, h = faces[0]
        color_hex = shape_info[face_shape]['color']
        color_bgr = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

        cv2.rectangle(img_out, (x, y), (x+w, y+h), color_bgr, 3)
        for pt in [(x+w//2, y), (x+w//2, y+h), (x, y+h//2), (x+w, y+h//2)]:
            cv2.circle(img_out, pt, 7, color_bgr, -1)
            cv2.circle(img_out, pt, 7, (255,255,255), 2)

        face_height = float(h)
        face_width  = float(w)
        if face_width > 0:
            ratiog = face_height / face_width
            score  = max(0.0, min((1 - abs(ratiog - 1.618) / 1.618) * 100, 100))

    return face_shape, confidence, ratiog, score, img_out, face_detected


if uploaded_file is not None:
    img_pil = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        with st.spinner("🔍 กำลังวิเคราะห์..."):
            face_shape, confidence, ratiog, score, img_out, face_detected = predict_face_shape(img_pil)

        st.image(img_out, use_column_width=True, caption="ผลการตรวจจับใบหน้า")

    with col2:
        if not face_detected:
            st.error("❌ ไม่พบใบหน้าในภาพ กรุณาลองใหม่")
        else:
            info = shape_info[face_shape]
            st.markdown(f"""
            <div class="result-card">
                <div style="color:#a0aec0; font-size:0.85rem; margin-bottom:0.3rem;">รูปทรงใบหน้าของคุณ</div>
                <div class="shape-badge" style="background:{info['color']}">
                    {info['emoji']} {face_shape}
                </div>
                <div style="color:#cbd5e0; font-size:0.88rem; margin-top:0.5rem;">{info['desc']}</div>

                <div class="metric-row">
                    <div class="metric-box">
                        <div class="val">{confidence:.1f}%</div>
                        <div class="lbl">ความมั่นใจ</div>
                    </div>
                    <div class="metric-box">
                        <div class="val">{ratiog:.2f}</div>
                        <div class="lbl">Golden Ratio</div>
                    </div>
                    <div class="metric-box">
                        <div class="val">{score:.0f}%</div>
                        <div class="lbl">คะแนนสัดส่วน</div>
                    </div>
                </div>

                <div style="color:#a0aec0; font-size:0.82rem; margin-top:1.2rem; font-weight:600;">
                    💇 ทรงผมที่แนะนำ
                </div>
                <div class="hair-section">{info['hair']}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div class="footer">Powered by InceptionResNetV2 + OpenCV · Face Shape AI</div>', unsafe_allow_html=True)
