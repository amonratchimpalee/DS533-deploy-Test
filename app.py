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
    'Oval':   {
        'emoji': '🥚',
        'color': [76,175,80],
        'gradient': 'linear-gradient(135deg, #11998e, #38ef7d)',
        'accent': '#38ef7d',
        'desc': 'ใบหน้ารูปไข่ — สมดุลที่สุด เหมาะกับทุกทรงผม',
        'hair': 'ผมสั้นถึงกลาง เช่น blunt bob, shoulder-length, pixie cut, long layers และหน้าม้าปัดข้าง',
    },
    'Square': {
        'emoji': '⬛',
        'color': [33,150,243],
        'gradient': 'linear-gradient(135deg, #2980b9, #6dd5fa)',
        'accent': '#6dd5fa',
        'desc': 'ใบหน้าเหลี่ยม — กรามและหน้าผากกว้างพอกัน',
        'hair': 'ผมยาวปานกลางถึงยาว พร้อมไล่เลเยอร์หรือปลายฟุ้ง เช่น beach waves และหน้าม้านุ่มๆ',
    },
    'Round':  {
        'emoji': '⭕',
        'color': [255,152,0],
        'gradient': 'linear-gradient(135deg, #f7971e, #ffd200)',
        'accent': '#ffd200',
        'desc': 'ใบหน้ากลม — แก้มอิ่ม ใบหน้ากว้างและสั้น',
        'hair': 'ทรงเพิ่มความสูงให้ใบหน้า เช่น textured bob, long layers, แสกข้าง และ blunt bangs',
    },
    'Heart':  {
        'emoji': '❤️',
        'color': [233,30,99],
        'gradient': 'linear-gradient(135deg, #c94b4b, #e96d8a)',
        'accent': '#ff6b9d',
        'desc': 'ใบหน้ารูปหัวใจ — หน้าผากกว้าง คางแหลม',
        'hair': 'ผมยาวระดับไหล่ พร้อมเลเยอร์บริเวณกราม curtain bangs หรือ wispy bangs',
    },
    'Oblong': {
        'emoji': '📏',
        'color': [156,39,176],
        'gradient': 'linear-gradient(135deg, #834d9b, #d04ed6)',
        'accent': '#d04ed6',
        'desc': 'ใบหน้ายาว — ยาวกว่ากว้างมาก',
        'hair': 'ลอนคลาย, loose curls, layered bob และหน้าม้าปัดข้างหรือ curtain bangs',
    },
}

# ── Page config ──
st.set_page_config(page_title="Face Shape AI ✨", page_icon="✨", layout="centered")

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(120, 40, 200, 0.25) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255, 100, 150, 0.18) 0%, transparent 60%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 2rem; }

h1, h2, h3, p, span, div, label {
    font-family: 'DM Sans', sans-serif !important;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #fff 0%, #e8b4f0 40%, #f5a623 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    text-align: center;
    color: rgba(255,255,255,0.35);
    font-size: 0.9rem;
    font-weight: 300;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), rgba(245,166,35,0.35), rgba(255,255,255,0.12), transparent);
    margin: 1.5rem 0 2rem;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(255,255,255,0.15) !important;
    border-radius: 20px !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(245,166,35,0.45) !important;
    background: rgba(245,166,35,0.04) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: rgba(255,255,255,0.45) !important;
}

.stImage img { border-radius: 18px; border: 1px solid rgba(255,255,255,0.1); }

[data-testid="stAlert"] {
    background: rgba(233,30,99,0.1) !important;
    border: 1px solid rgba(233,30,99,0.3) !important;
    border-radius: 14px !important;
    color: #ff6b9d !important;
}
.footer {
    text-align: center;
    color: rgba(255,255,255,0.15);
    font-size: 0.75rem;
    padding: 2.5rem 0 1rem;
    letter-spacing: 0.05em;
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Header ──
st.markdown("<div class='hero-title'>✨ Face Shape AI</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>วิเคราะห์รูปใบหน้าและแนะนำทรงผมด้วย AI</div>", unsafe_allow_html=True)
st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

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
            info = shape_info[face_shape]
            gradient = info['gradient']
            accent   = info['accent']
            emoji    = info['emoji']
            desc     = info['desc']
            hair     = info['hair']
            conf_str = f"{confidence:.1f}"
            ratio_str = f"{ratiog:.2f}"
            score_str = f"{score:.0f}"

            card_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; font-family: 'DM Sans', sans-serif; }}

  .card {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    padding: 1.6rem;
    position: relative;
    overflow: hidden;
  }}
  .card::before {{
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 24px;
    padding: 1.5px;
    background: {gradient};
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    opacity: 0.7;
    pointer-events: none;
  }}

  .emoji {{ font-size: 2.2rem; margin-bottom: .3rem; }}
  .shape-name {{
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: #fff;
    line-height: 1.1;
    margin-bottom: .3rem;
  }}
  .desc {{
    color: rgba(255,255,255,0.45);
    font-size: 0.85rem;
    line-height: 1.6;
    margin-bottom: 1rem;
  }}

  .conf-label {{
    font-size: 0.68rem;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: .25rem;
  }}
  .conf-value {{
    font-size: 1.7rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: .4rem;
  }}
  .bar-bg {{
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    height: 5px;
    overflow: hidden;
    margin-bottom: 1.1rem;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 99px;
    background: {gradient};
    width: {conf_str}%;
  }}

  .metrics {{
    display: flex;
    gap: .6rem;
    margin-bottom: 1.1rem;
  }}
  .metric {{
    flex: 1;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: .85rem .5rem;
    text-align: center;
  }}
  .m-icon {{ font-size: 1rem; margin-bottom: .15rem; }}
  .m-val {{ font-size: 1.35rem; font-weight: 700; color: #fff; line-height: 1; }}
  .m-label {{ font-size: 0.65rem; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: .06em; margin-top: .2rem; }}

  .hair-box {{
    background: rgba(255,255,255,0.04);
    border-left: 3px solid {accent};
    border-radius: 0 12px 12px 0;
    padding: .9rem 1rem;
  }}
  .hair-title {{
    color: {accent};
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: .35rem;
  }}
  .hair-text {{
    color: rgba(255,255,255,0.7);
    font-size: 0.87rem;
    line-height: 1.6;
  }}
</style>
</head>
<body>
<div class="card">
  <div class="emoji">{emoji}</div>
  <div class="shape-name">{face_shape}</div>
  <div class="desc">{desc}</div>

  <div class="conf-label">ความมั่นใจ</div>
  <div class="conf-value">{conf_str}%</div>
  <div class="bar-bg"><div class="bar-fill"></div></div>

  <div class="metrics">
    <div class="metric">
      <div class="m-icon">📐</div>
      <div class="m-val">{ratio_str}</div>
      <div class="m-label">Golden Ratio</div>
    </div>
    <div class="metric">
      <div class="m-icon">⭐</div>
      <div class="m-val">{score_str}%</div>
      <div class="m-label">คะแนน</div>
    </div>
  </div>

  <div class="hair-box">
    <div class="hair-title">💇 ทรงผมที่แนะนำ</div>
    <div class="hair-text">{hair}</div>
  </div>
</div>
</body>
</html>
"""
            components.html(card_html, height=480, scrolling=False)

st.markdown("<div class='footer'>Powered by 4 angie · OpenCV</div>", unsafe_allow_html=True)
