import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import gdown
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import mediapipe as mp  # ใช้เวอร์ชัน PyPI ≥0.10

# ---------- Model preprocessing ----------
@keras.saving.register_keras_serializable()
def preprocess(x):
    x = tf.cast(x, tf.float32)
    return preprocess_input(x)

MODEL_URL   = "https://drive.google.com/uc?id=1p3veX7I7_6WBM97jOSfQpSGcxwIuijD1"
MODEL_LOCAL = "best_inceptionresnetv2_face_shape_fixed.keras"

@st.cache_resource
def load_models():
    if not tf.io.gfile.exists(MODEL_LOCAL):
        gdown.download(MODEL_URL, MODEL_LOCAL, quiet=False)
    face_model = tf.keras.models.load_model(MODEL_LOCAL, custom_objects={'preprocess': preprocess})
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_model, face_cascade

# ---------- Face Shape Classes ----------
classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
shape_info = {
    'Oval':   {'emoji':'🥚','color':[218,165,32],'desc':'ใบหน้ารูปไข่ — สมดุลที่สุด เหมาะกับทุกทรงผม','hair':'ผมสั้นถึงกลาง blunt bob, shoulder-length, pixie cut, long layers และหน้าม้าปัดข้าง'},
    'Square': {'emoji':'⬛','color':[210,140,0],'desc':'ใบหน้าเหลี่ยม — กรามและหน้าผากกว้างพอกัน','hair':'ผมยาวปานกลางถึงยาว พร้อมไล่เลเยอร์หรือปลายฟุ้ง beach waves, หน้าม้านุ่มๆ'},
    'Round':  {'emoji':'⭕','color':[232,120,0],'desc':'ใบหน้ากลม — แก้มอิ่ม ใบหน้ากว้างและสั้น','hair':'ทรงเพิ่มความสูงให้ใบหน้า textured bob, long layers, แสกข้าง, blunt bangs'},
    'Heart':  {'emoji':'❤️','color':[200,150,0],'desc':'ใบหน้ารูปหัวใจ — หน้าผากกว้าง คางแหลม','hair':'ผมยาวระดับไหล่ เลเยอร์บริเวณกราม curtain bangs, wispy bangs'},
    'Oblong': {'emoji':'📏','color':[180,120,0],'desc':'ใบหน้ายาว — ยาวกว่ากว้างมาก','hair':'ลอนคลาย, loose curls, layered bob, หน้าม้าปัดข้างหรือ curtain bangs'},
}

# ---------- Streamlit Page ----------
st.set_page_config(page_title="Face Shape AI ✨", page_icon="✨", layout="centered")
st.title("✨ Face Shape AI")
st.subheader("วิเคราะห์รูปใบหน้าและแนะนำทรงผมด้วย AI")

face_shape_model, face_cascade = load_models()
uploaded_file = st.file_uploader("📸  อัปโหลดภาพใบหน้าของคุณ", type=["jpg","jpeg","png"])

# ---------- Mediapipe FaceMesh ----------
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ---------- Landmark Detection ----------
def find_landmarks(img_rgb, face_rect):
    ih, iw = img_rgb.shape[:2]
    results = face_mesh.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        tr = (int(lm[10].x*iw), int(lm[10].y*ih))     # Hairline
        gn = (int(lm[152].x*iw), int(lm[152].y*ih))   # Chin
        zy_l = (int(lm[234].x*iw), int(lm[234].y*ih)) # Left Zygion
        zy_r = (int(lm[454].x*iw), int(lm[454].y*ih)) # Right Zygion
        face_h_px = gn[1] - tr[1]
        face_w_meas = zy_r[0] - zy_l[0]
        return tr, gn, zy_l, zy_r, face_h_px, face_w_meas
    else:
        x, y, w, h = face_rect
        tr = (x+w//2, y)
        gn = (x+w//2, y+h)
        zy_l = (x, y+h//2)
        zy_r = (x+w, y+h//2)
        return tr, gn, zy_l, zy_r, h, w

def draw_landmarks(img_out, tr, gn, zy_l, zy_r, color_bgr):
    cv2.line(img_out, ((tr[0]+gn[0])//2, tr[1]), ((tr[0]+gn[0])//2, gn[1]), color_bgr, 2)
    cv2.line(img_out, zy_l, zy_r, color_bgr, 2)
    for pt in [tr, gn, zy_l, zy_r]:
        cv2.circle(img_out, pt, 5, color_bgr, -1)

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
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    if len(faces)>0:
        face_detected=True
        x,y,w,h = faces[0]
        tr, gn, zy_l, zy_r, face_h, face_w = find_landmarks(img, (x,y,w,h))
        draw_landmarks(img_out, tr, gn, zy_l, zy_r, (0,255,0))
        ratiog = face_h/face_w if face_w>0 else 1
        score = max(0.0, min((1 - abs(ratiog-1.6)/1.6)*100, 100))
    return face_shape, confidence, ratiog, score, img_out, face_detected

# ---------- Streamlit App ----------
if uploaded_file:
    img_pil = Image.open(uploaded_file)
    face_shape, confidence, ratiog, score, img_out, face_detected = predict_face_shape(img_pil)
    st.image(img_out, use_column_width=True)
    if face_detected:
        info = shape_info[face_shape]
        st.markdown(f"**{info['emoji']} {face_shape}**")
        st.markdown(f"{info['desc']}")
        st.markdown(f"**Confidence:** {confidence:.1f}%")
        st.markdown(f"**Facial Index:** {ratiog:.2f} (Score near Phi: {score:.0f}%)")
        st.markdown(f"**Suggested Hair:** {info['hair']}")
    else:
        st.warning("ไม่พบใบหน้าในภาพ กรุณาลองภาพอื่น")
