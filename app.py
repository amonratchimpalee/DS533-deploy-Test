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

MODEL_URL   = "https://drive.google.com/uc?id=1p3veX7I7_6WBM97jOSfQpSGcxwIuijD1"
MODEL_LOCAL = "best_inceptionresnetv2_face_shape_fixed.keras"

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_LOCAL):
        gdown.download(MODEL_URL, MODEL_LOCAL, quiet=False)
    face_model = tf.keras.models.load_model(MODEL_LOCAL, custom_objects={'preprocess': preprocess})
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_model, face_cascade

classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

shape_info = {
    'Oval':   {'emoji': '🥚', 'color': [76,175,80],   'desc': 'ใบหน้ารูปไข่ — สมดุลที่สุด เหมาะกับทุกทรงผม',
               'hair': 'ผมสั้นถึงกลาง เช่น blunt bob, shoulder-length, pixie cut, long layers และหน้าม้าปัดข้าง'},
    'Square': {'emoji': '⬛', 'color': [33,150,243],  'desc': 'ใบหน้าเหลี่ยม — กรามและหน้าผากกว้างพอกัน',
               'hair': 'ผมยาวปานกลางถึงยาว พร้อมไล่เลเยอร์หรือปลายฟุ้ง เช่น beach waves และหน้าม้านุ่มๆ'},
    'Round':  {'emoji': '⭕', 'color': [255,152,0],   'desc': 'ใบหน้ากลม — แก้มอิ่ม ใบหน้ากว้างและสั้น',
               'hair': 'ทรงเพิ่มความสูงให้ใบหน้า เช่น textured bob, long layers, แสกข้าง และ blunt bangs'},
    'Heart':  {'emoji': '❤️', 'color': [233,30,99],   'desc': 'ใบหน้ารูปหัวใจ — หน้าผากกว้าง คางแหลม',
               'hair': 'ผมยาวระดับไหล่ พร้อมเลเยอร์บริเวณกราม curtain bangs หรือ wispy bangs'},
    'Oblong': {'emoji': '📏', 'color': [156,39,176],  'desc': 'ใบหน้ายาว — ยาวกว่ากว้างมาก',
               'hair': 'ลอนคลาย, loose curls, layered bob และหน้าม้าปัดข้างหรือ curtain bangs'},
}

# ── Page config ──
st.set_page_config(page_title="Face Shape AI ✨", page_icon="✨", layout="centered")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("<h1 style='text-align:center;color:#f5a623;'>✨ Face Shape AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#a0aec0;'>วิเคราะห์รูปใบหน้าและแนะนำทรงผมด้วย AI</p>", unsafe_allow_html=True)
st.divider()

face_shape_model, face_cascade = load_models()

uploaded_file = st.file_uploader("📸 อัปโหลดภาพใบหน้า", type=["jpg","jpeg","png"])

os.makedirs("saved_results", exist_ok=True)

def predict_face_shape(img_pil):
    img     = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_out = img.copy()

    img_resized = cv2.resize(img_bgr, (299, 299))
    pred        = face_shape_model.predict(np.expand_dims(img_resized, 0), verbose=0)
    idx         = np.argmax(pred)
    face_shape  = classes[idx]
    confidence  = float(pred[0][idx]) * 100

    ratiog, score, face_detected = 0.0, 0.0, False
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

    if len(faces) > 0:
        face_detected = True
        x, y, w, h = faces[0]
        c = tuple(shape_info[face_shape]['color'][::-1])  # BGR
        cv2.rectangle(img_out, (x,y), (x+w,y+h), c, 3)
        for pt in [(x+w//2,y),(x+w//2,y+h),(x,y+h//2),(x+w,y+h//2)]:
            cv2.circle(img_out, pt, 8, c, -1)
            cv2.circle(img_out, pt, 8, (255,255,255), 2)
        ratiog = h / w if w > 0 else 0
        score  = max(0, min((1 - abs(ratiog-1.618)/1.618)*100, 100))

    return face_shape, confidence, ratiog, score, img_out, face_detected


if uploaded_file:
    img_pil = Image.open(uploaded_file)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.spinner("🔍 กำลังวิเคราะห์..."):
            face_shape, confidence, ratiog, score, img_out, face_detected = predict_face_shape(img_pil)
        st.image(img_out, use_column_width=True)

    with col2:
        if not face_detected:
            st.error("❌ ไม่พบใบหน้าในภาพ กรุณาลองใหม่")
        else:
            info = shape_info[face_shape]
            st.markdown(f"### {info['emoji']} รูปทรงใบหน้า: **{face_shape}**")
            st.caption(info['desc'])
            st.divider()

            m1, m2, m3 = st.columns(3)
            m1.metric("🎯 ความมั่นใจ", f"{confidence:.1f}%")
            m2.metric("📐 Golden Ratio", f"{ratiog:.2f}")
            m3.metric("⭐ คะแนน", f"{score:.0f}%")

            st.divider()
            st.markdown("**💇 ทรงผมที่แนะนำ**")
            st.info(info['hair'])

st.markdown("<p style='text-align:center;color:#4a5568;font-size:0.8rem;padding-top:2rem;'>Powered by InceptionResNetV2 + OpenCV</p>", unsafe_allow_html=True)
