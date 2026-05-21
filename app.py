import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import tensorflow as tf
import keras
import os
from PIL import Image
import gdown
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import mediapipe as mp

# ---------- Model preprocessing ----------
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

# ---------- Face Shape Classes ----------
classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

shape_info = {
    'Oval':   {'emoji':'🥚','color':[218,165,32],'gradient':'linear-gradient(135deg,#f7c948,#ffe08a)','accent':'#ffe08a',
               'desc':'ใบหน้ารูปไข่ — สมดุลที่สุด เหมาะกับทุกทรงผม',
               'hair':'ผมสั้นถึงกลาง เช่น blunt bob, shoulder-length, pixie cut, long layers และหน้าม้าปัดข้าง'},
    'Square': {'emoji':'⬛','color':[210,140,0],'gradient':'linear-gradient(135deg,#d48c00,#f5c842)','accent':'#f5c842',
               'desc':'ใบหน้าเหลี่ยม — กรามและหน้าผากกว้างพอกัน',
               'hair':'ผมยาวปานกลางถึงยาว พร้อมไล่เลเยอร์หรือปลายฟุ้ง เช่น beach waves และหน้าม้านุ่มๆ'},
    'Round':  {'emoji':'⭕','color':[232,120,0],'gradient':'linear-gradient(135deg,#e87800,#ffc13b)','accent':'#ffc13b',
               'desc':'ใบหน้ากลม — แก้มอิ่ม ใบหน้ากว้างและสั้น',
               'hair':'ทรงเพิ่มความสูงให้ใบหน้า เช่น textured bob, long layers, แสกข้าง และ blunt bangs'},
    'Heart':  {'emoji':'❤️','color':[200,150,0],'gradient':'linear-gradient(135deg,#c89600,#fada5e)','accent':'#fada5e',
               'desc':'ใบหน้ารูปหัวใจ — หน้าผากกว้าง คางแหลม',
               'hair':'ผมยาวระดับไหล่ พร้อมเลเยอร์บริเวณกราม curtain bangs หรือ wispy bangs'},
    'Oblong': {'emoji':'📏','color':[180,120,0],'gradient':'linear-gradient(135deg,#b47800,#f0b429)','accent':'#f0b429',
               'desc':'ใบหน้ายาว — ยาวกว่ากว้างมาก',
               'hair':'ลอนคลาย, loose curls, layered bob และหน้าม้าปัดข้างหรือ curtain bangs'},
}

# ---------- Mediapipe setup ----------
mp_face_mesh = mp.solutions.face_mesh

# ---------- Streamlit Page ----------
st.set_page_config(page_title="Face Shape AI ✨", page_icon="✨", layout="centered")
st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">✨ Face Shape AI</div>
  <div class="hero-sub">วิเคราะห์รูปใบหน้าและแนะนำทรงผมด้วย AI</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

face_shape_model, face_cascade = load_models()
uploaded_file = st.file_uploader("📸  อัปโหลดภาพใบหน้าของคุณ", type=["jpg","jpeg","png"])
os.makedirs("saved_results", exist_ok=True)

# ---------- Landmark Detection ----------
def find_landmarks(img_rgb, face_rect):
    ih, iw = img_rgb.shape[:2]

    with mp_face_mesh.FaceMesh(static_image_mode=True,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # trichion (บน midline)
        tr_x = int(lm[10].x * iw)
        tr_y = int(lm[10].y * ih)

        # gnathion (คาง)
        gn_x = int(lm[152].x * iw)
        gn_y = int(lm[152].y * ih)

        # zygion ซ้าย/ขวา (จุดโหนกแก้ม)
        zy_l_x = int(lm[234].x * iw)
        zy_l_y = int(lm[234].y * ih)
        zy_r_x = int(lm[454].x * iw)
        zy_r_y = int(lm[454].y * ih)

        tr_abs = (tr_x, tr_y)
        gn_abs = (gn_x, gn_y)
        zy_l = (zy_l_x, zy_l_y)
        zy_r = (zy_r_x, zy_r_y)

        face_h_px = gn_y - tr_y
        face_w_meas = zy_r_x - zy_l_x

        return tr_abs, gn_abs, zy_l, zy_r, face_h_px, face_w_meas
    else:
        x, y, w, h = face_rect
        tr_abs = (x + w//2, y)
        gn_abs = (x + w//2, y + h)
        zy_l = (x, y + h//2)
        zy_r = (x + w, y + h//2)
        return tr_abs, gn_abs, zy_l, zy_r, h, w

# ---------- Draw Landmarks ----------
def draw_landmarks(img_out, tr, gn, zy_l, zy_r, color_bgr):
    c  = color_bgr
    cw = (255, 255, 255)

    mid_x = (tr[0] + gn[0]) // 2
    cv2.line(img_out, (mid_x, tr[1]), (mid_x, gn[1]), c,  2, cv2.LINE_AA)
    cv2.line(img_out, zy_l, zy_r, c, 2, cv2.LINE_AA)

    landmarks = [(tr, "A (tr)"), (gn, "B (gn)"), (zy_l, "C (zy)"), (zy_r, "D (zy)")]
    for pt, lbl in landmarks:
        cv2.circle(img_out, pt, 7, c, -1, cv2.LINE_AA)
        cv2.circle(img_out, pt, 7, cw, 2, cv2.LINE_AA)
        tx = pt[0] + 10 if pt[0] < img_out.shape[1] - 60 else pt[0] - 70
        ty = pt[1] - 8 if pt[1] > 20 else pt[1] + 18
        cv2.putText(img_out, lbl, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cw, 1, cv2.LINE_AA)

# ---------- Predict Face Shape ----------
def predict_face_shape(img_pil):
    img = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_out = img.copy()

    img_resized = cv2.resize(img_bgr, (299, 299))
    pred = face_shape_model.predict(np.expand_dims(img_resized, 0), verbose=0)
    idx = np.argmax(pred)
    face_shape = classes[idx]
    confidence = float(pred[0][idx]) * 100

    ratiog, score, face_detected = 0.0, 0.0, False

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) > 0:
        face_detected = True
        x, y, w, h = faces[0]
        c_bgr = tuple(shape_info[face_shape]['color'][::-1])
        cv2.rectangle(img_out, (x, y), (x+w, y+h), c_bgr, 1)

        tr, gn, zy_l, zy_r, face_h_meas, face_w_meas = find_landmarks(img, (x, y, w, h))
        draw_landmarks(img_out, tr, gn, zy_l, zy_r, c_bgr)

        if face_w_meas > 0:
            ratiog = face_h_meas / face_w_meas
        else:
            ratiog = 1.0

        score = max(0.0, min((1 - abs(ratiog - 1.6) / 1.6) * 100, 100))
        label = f"Facial Index: {ratiog:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx, ly = x, max(y - 12, th + 4)
        cv2.rectangle(img_out, (lx-2, ly-th-4), (lx+tw+4, ly+4), (0,0,0), -1)
        cv2.putText(img_out, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_bgr, 1, cv2.LINE_AA)

    return face_shape, confidence, ratiog, score, img_out, face_detected

# ---------- Streamlit App ----------
if uploaded_file:
    img_pil = Image.open(uploaded_file)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.spinner("🔍 กำลังวิเคราะห์..."):
            face_shape, confidence, ratiog, score, img_out, face_detected = predict_face_shape(img_pil)
        st.image(img_out, use_container_width=True)

    with col2:
        if not face_detected:
            st.error("❌ ไม่พบใบหน้าในภาพ กรุณาลองภาพอื่น")
        else:
            info      = shape_info[face_shape]
            gradient  = info['gradient']
            accent    = info['accent']
            emoji     = info['emoji']
            desc      = info['desc']
            hair      = info['hair']
            conf_str  = f"{confidence:.1f}"
            ratio_str = f"{ratiog:.2f}"
            score_str = f"{score:.0f}"

            if ratiog >= 1.55 and ratiog <= 1.65:
                fi_label = "Normal (≈ 1.6)"
            elif ratiog > 1.65:
                fi_label = "Long face (> 1.6)"
            else:
                fi_label = "Short face (< 1.6)"

            # HTML Card Rendering (เหมือนเดิม)
            card_html = f"<div class='card'>...</div>"
            components.html(card_html, height=510, scrolling=False)

st.markdown("<div class='footer'>Powered by <b>4 angie</b> · อ้างอิง: Saraswathi (2007) Eur J Anat 11(3):177-180</div>",
            unsafe_allow_html=True)
