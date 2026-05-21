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
    'Oval':   {'emoji':'🥚','color':[76,175,80],  'gradient':'linear-gradient(135deg,#11998e,#38ef7d)','accent':'#38ef7d',
               'desc':'ใบหน้ารูปไข่ — สมดุลที่สุด เหมาะกับทุกทรงผม',
               'hair':'ผมสั้นถึงกลาง เช่น blunt bob, shoulder-length, pixie cut, long layers และหน้าม้าปัดข้าง'},
    'Square': {'emoji':'⬛','color':[33,150,243], 'gradient':'linear-gradient(135deg,#2980b9,#6dd5fa)','accent':'#6dd5fa',
               'desc':'ใบหน้าเหลี่ยม — กรามและหน้าผากกว้างพอกัน',
               'hair':'ผมยาวปานกลางถึงยาว พร้อมไล่เลเยอร์หรือปลายฟุ้ง เช่น beach waves และหน้าม้านุ่มๆ'},
    'Round':  {'emoji':'⭕','color':[255,152,0],  'gradient':'linear-gradient(135deg,#f7971e,#ffd200)','accent':'#ffd200',
               'desc':'ใบหน้ากลม — แก้มอิ่ม ใบหน้ากว้างและสั้น',
               'hair':'ทรงเพิ่มความสูงให้ใบหน้า เช่น textured bob, long layers, แสกข้าง และ blunt bangs'},
    'Heart':  {'emoji':'❤️','color':[233,30,99],  'gradient':'linear-gradient(135deg,#c94b4b,#e96d8a)','accent':'#ff6b9d',
               'desc':'ใบหน้ารูปหัวใจ — หน้าผากกว้าง คางแหลม',
               'hair':'ผมยาวระดับไหล่ พร้อมเลเยอร์บริเวณกราม curtain bangs หรือ wispy bangs'},
    'Oblong': {'emoji':'📏','color':[156,39,176], 'gradient':'linear-gradient(135deg,#834d9b,#d04ed6)','accent':'#d04ed6',
               'desc':'ใบหน้ายาว — ยาวกว่ากว้างมาก',
               'hair':'ลอนคลาย, loose curls, layered bob และหน้าม้าปัดข้างหรือ curtain bangs'},
}

st.set_page_config(page_title="Face Shape AI ✨", page_icon="✨", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Background ── */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
.main, .block-container {
    background: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 50% at 20% -10%, rgba(120,40,200,.28) 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,100,150,.2) 0%, transparent 60%),
                #09090f !important;
    min-height: 100vh;
}
[data-testid="stHeader"]        { background: transparent !important; }
[data-testid="stToolbar"]       { background: transparent !important; }
[data-testid="stDecoration"]    { display: none !important; }
.main .block-container          { padding-top: 2.5rem !important; max-width: 780px; }

/* ── Global font ── */
html, body, [class*="css"], [data-testid], p, span, div, label, button {
    font-family: 'DM Sans', sans-serif !important;
    color: rgba(255,255,255,.85);
}

/* ── Hero ── */
.hero-wrap { text-align: center; margin-bottom: 1.8rem; }
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2rem, 6vw, 3.2rem);
    font-weight: 900 !important;
    background: linear-gradient(135deg, #fff 0%, #ddb4f8 45%, #f5a623 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -.02em;
    line-height: 1.1;
    margin-bottom: .4rem;
}
.hero-sub {
    color: rgba(255,255,255,.3) !important;
    font-size: .85rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    -webkit-text-fill-color: rgba(255,255,255,.3) !important;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.1), rgba(245,166,35,.4), rgba(255,255,255,.1), transparent);
    margin: 0 0 2rem;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,.04) !important;
    border: 1.5px dashed rgba(255,255,255,.18) !important;
    border-radius: 18px !important;
}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(245,166,35,.5) !important;
    background: rgba(245,166,35,.04) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: rgba(255,255,255,.4) !important;
    -webkit-text-fill-color: rgba(255,255,255,.4) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label * {
    color: rgba(255,255,255,.7) !important;
    -webkit-text-fill-color: rgba(255,255,255,.7) !important;
    font-size: .95rem !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(255,255,255,.08) !important;
    border: 1px solid rgba(255,255,255,.2) !important;
    border-radius: 8px !important;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    box-shadow: 0 20px 60px rgba(0,0,0,.5) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] * { color: rgba(255,255,255,.5) !important; }

/* ── Error ── */
[data-testid="stAlert"] {
    background: rgba(233,30,99,.1) !important;
    border: 1px solid rgba(233,30,99,.3) !important;
    border-radius: 14px !important;
}
[data-testid="stAlert"] * { color: #ff6b9d !important; -webkit-text-fill-color: #ff6b9d !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    font-size: .75rem;
    color: rgba(255,255,255,.15) !important;
    -webkit-text-fill-color: rgba(255,255,255,.15) !important;
    letter-spacing: .06em;
}
.footer b { color: rgba(245,166,35,.45) !important; -webkit-text-fill-color: rgba(245,166,35,.45) !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
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
        c = tuple(shape_info[face_shape]['color'][::-1])
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

            card_html = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<link href='https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap' rel='stylesheet'>"
                "<style>"
                "*{box-sizing:border-box;margin:0;padding:0;}"
                "body{background:transparent;font-family:'DM Sans',sans-serif;}"
                ".card{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);"
                "border-radius:24px;padding:1.5rem;position:relative;overflow:hidden;}"
                ".card::before{content:'';position:absolute;inset:0;border-radius:24px;padding:1.5px;"
                "background:" + gradient + ";"
                "-webkit-mask:linear-gradient(#fff 0 0) content-box,linear-gradient(#fff 0 0);"
                "-webkit-mask-composite:xor;mask-composite:exclude;opacity:.7;pointer-events:none;}"
                ".emoji{font-size:2.2rem;margin-bottom:.3rem;}"
                ".shape-name{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:900;"
                "color:#fff;line-height:1.1;margin-bottom:.3rem;}"
                ".desc{color:rgba(255,255,255,.4);font-size:.83rem;line-height:1.6;margin-bottom:.9rem;}"
                ".conf-label{font-size:.65rem;color:rgba(255,255,255,.28);text-transform:uppercase;"
                "letter-spacing:.09em;margin-bottom:.2rem;}"
                ".conf-value{font-size:1.8rem;font-weight:700;color:#fff;margin-bottom:.35rem;}"
                ".bar-bg{background:rgba(255,255,255,.08);border-radius:99px;height:5px;overflow:hidden;margin-bottom:1rem;}"
                ".bar-fill{height:100%;border-radius:99px;background:" + gradient + ";width:" + conf_str + "%;}"
                ".metrics{display:flex;gap:.55rem;margin-bottom:1rem;}"
                ".metric{flex:1;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);"
                "border-radius:13px;padding:.8rem .4rem;text-align:center;}"
                ".m-icon{font-size:.95rem;margin-bottom:.12rem;}"
                ".m-val{font-size:1.3rem;font-weight:700;color:#fff;line-height:1;}"
                ".m-label{font-size:.62rem;color:rgba(255,255,255,.28);text-transform:uppercase;"
                "letter-spacing:.06em;margin-top:.18rem;}"
                ".hair-box{background:rgba(255,255,255,.04);border-left:3px solid " + accent + ";"
                "border-radius:0 12px 12px 0;padding:.85rem 1rem;}"
                ".hair-title{color:" + accent + ";font-size:.68rem;font-weight:600;"
                "text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem;}"
                ".hair-text{color:rgba(255,255,255,.68);font-size:.85rem;line-height:1.6;}"
                "</style></head><body>"
                "<div class='card'>"
                "<div class='emoji'>" + emoji + "</div>"
                "<div class='shape-name'>" + face_shape + "</div>"
                "<div class='desc'>" + desc + "</div>"
                "<div class='conf-label'>ความมั่นใจ</div>"
                "<div class='conf-value'>" + conf_str + "%</div>"
                "<div class='bar-bg'><div class='bar-fill'></div></div>"
                "<div class='metrics'>"
                "<div class='metric'><div class='m-icon'>📐</div><div class='m-val'>" + ratio_str + "</div><div class='m-label'>Golden Ratio</div></div>"
                "<div class='metric'><div class='m-icon'>⭐</div><div class='m-val'>" + score_str + "%</div><div class='m-label'>คะแนน</div></div>"
                "</div>"
                "<div class='hair-box'>"
                "<div class='hair-title'>💇 ทรงผมที่แนะนำ</div>"
                "<div class='hair-text'>" + hair + "</div>"
                "</div>"
                "</div>"
                "</body></html>"
            )
            components.html(card_html, height=470, scrolling=False)

st.markdown("<div class='footer'>Powered by <b>InceptionResNetV2</b> · <b>OpenCV</b></div>", unsafe_allow_html=True)
