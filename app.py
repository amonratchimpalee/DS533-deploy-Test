# =========================================================
# IMPORTS
# =========================================================

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


# =========================================================
# PREPROCESS
# =========================================================

@keras.saving.register_keras_serializable()
def preprocess(x):
    x = tf.cast(x, tf.float32)
    return preprocess_input(x)


# =========================================================
# MODEL
# =========================================================

MODEL_URL   = "https://drive.google.com/uc?id=1p3veX7I7_6WBM97jOSfQpSGcxwIuijD1"
MODEL_LOCAL = "best_inceptionresnetv2_face_shape_fixed.keras"


@st.cache_resource
def load_models():

    if not os.path.exists(MODEL_LOCAL):
        gdown.download(MODEL_URL, MODEL_LOCAL, quiet=False)

    face_model = tf.keras.models.load_model(
        MODEL_LOCAL,
        custom_objects={'preprocess': preprocess}
    )

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import urllib.request

    model_path = "face_landmarker.task"

    if not os.path.exists(model_path):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            model_path
        )

    base_options = mp_python.BaseOptions(model_asset_path=model_path)

    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    face_mesh = mp_vision.FaceLandmarker.create_from_options(options)

    return face_model, face_mesh


# =========================================================
# CONFIG
# =========================================================

classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

LANDMARK_GNATHION = 152
LANDMARK_ZY_LEFT  = 116
LANDMARK_ZY_RIGHT = 345


shape_info = {

    'Oval': {
        'emoji':'🥚',
        'color':[218,165,32],
        'gradient':'linear-gradient(135deg,#f7c948,#ffe08a)',
        'accent':'#ffe08a',
        'desc':'ใบหน้ารูปไข่',
        'hair':'ผมสั้นถึงกลาง เช่น blunt bob, shoulder-length, pixie cut, long layers และหน้าม้าปัดข้าง',
        'glasses':'ทุกทรงเหมาะกับใบหน้ารูปไข่ แนะนำ rectangle, square และ aviator เพื่อเพิ่มความคมชัด'
    },

    'Square': {
        'emoji':'⬛',
        'color':[210,140,0],
        'gradient':'linear-gradient(135deg,#d48c00,#f5c842)',
        'accent':'#f5c842',
        'desc':'ใบหน้าเหลี่ยม',
        'hair':'ผมยาวปานกลางถึงยาว พร้อมไล่เลเยอร์หรือปลายฟุ้ง เช่น beach waves และหน้าม้านุ่มๆ',
        'glasses':'แนะนำ round, oval และ aviator เพื่อลดความเหลี่ยม'
    },

    'Round': {
        'emoji':'⭕',
        'color':[232,120,0],
        'gradient':'linear-gradient(135deg,#e87800,#ffc13b)',
        'accent':'#ffc13b',
        'desc':'ใบหน้ากลม',
        'hair':'ทรงเพิ่มความสูงให้ใบหน้า เช่น textured bob, long layers, แสกข้าง และ blunt bangs',
        'glasses':'แนะนำ rectangle และ square เพื่อเพิ่มความยาวให้ใบหน้า'
    },

    'Heart': {
        'emoji':'❤️',
        'color':[200,150,0],
        'gradient':'linear-gradient(135deg,#c89600,#fada5e)',
        'accent':'#fada5e',
        'desc':'ใบหน้ารูปหัวใจ',
        'hair':'ผมยาวระดับไหล่ พร้อมเลเยอร์บริเวณกราม curtain bangs หรือ wispy bangs',
        'glasses':'แนะนำ oval, aviator และ rimless'
    },

    'Oblong': {
        'emoji':'📏',
        'color':[180,120,0],
        'gradient':'linear-gradient(135deg,#b47800,#f0b429)',
        'accent':'#f0b429',
        'desc':'ใบหน้ายาว',
        'hair':'ลอนคลาย, loose curls, layered bob และหน้าม้าปัดข้าง',
        'glasses':'แนะนำ square, round และ oversized'
    },
}


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Face Shape AI ✨",
    page_icon="✨",
    layout="centered"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"]>div,
.main,
.block-container{
    background:transparent!important;
}

[data-testid="stAppViewContainer"]{
    background:
    radial-gradient(ellipse 80% 50% at 20% -10%,rgba(200,130,0,.22) 0%,transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 110%,rgba(180,80,0,.18) 0%,transparent 60%),
    #0d0a04!important;
    min-height:100vh;
}

[data-testid="stHeader"],
[data-testid="stToolbar"]{
    background:transparent!important;
}

[data-testid="stDecoration"]{
    display:none!important;
}

.main .block-container{
    padding-top:2.5rem!important;
    max-width:780px;
}

html,body,[class*="css"],p,span,div,label,button{
    font-family:'DM Sans',sans-serif!important;
    color:rgba(255,255,255,.85);
}

.hero-wrap{
    text-align:center;
    margin-bottom:1.8rem;
}

.hero-title{
    font-family:'Playfair Display',serif!important;
    font-size:clamp(2rem,6vw,3.2rem);
    font-weight:900!important;

    background:linear-gradient(135deg,#fff 0%,#ffe8a0 45%,#e8860a 100%);

    -webkit-background-clip:text!important;
    -webkit-text-fill-color:transparent!important;

    line-height:1.1;
    margin-bottom:.4rem;
}

.hero-sub{
    color:rgba(255,255,255,.3)!important;
    font-size:.85rem;
}

.divider{
    height:1px;
    background:linear-gradient(
        90deg,
        transparent,
        rgba(255,255,255,.08),
        rgba(220,150,20,.6),
        rgba(255,255,255,.08),
        transparent
    );
    margin:0 0 2rem;
}

/* uploader */

[data-testid="stFileUploader"] label{
    display:none!important;
}

[data-testid="stFileUploader"] section{
    position:relative!important;

    background:rgba(255,255,255,.04)!important;

    border:1.5px dashed rgba(255,255,255,.18)!important;

    border-radius:18px!important;

    padding:0!important;

    min-height:120px!important;
}

[data-testid="stFileUploaderDropzoneInstructions"]{
    display:none!important;
}

[data-testid="stFileUploader"] section button{
    position:absolute!important;
    inset:0!important;

    width:100%!important;
    height:100%!important;

    opacity:0!important;
    cursor:pointer!important;
}

/* ซ่อนชื่อไฟล์ default */

[data-testid="stFileUploaderFile"]{
    display:none!important;
}

[data-testid="stFileUploaderFileName"]{
    display:none!important;
}

[data-testid="stFileUploaderDeleteBtn"]{
    display:none!important;
}

/* image */

[data-testid="stImage"] img{
    border-radius:18px!important;
    border:1px solid rgba(255,255,255,.1)!important;
    box-shadow:0 20px 60px rgba(0,0,0,.5)!important;
}

.footer{
    text-align:center;
    padding:2.5rem 0 1rem;
    font-size:.75rem;
    color:rgba(255,255,255,.15)!important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">✨ Face Shape classification</div>
    <div class="hero-sub">
        วิเคราะห์รูปใบหน้าและแนะนำทรงผมพร้อมแว่นตาที่เหมาะสม
    </div>
</div>

<div class="divider"></div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD
# =========================================================

face_shape_model, face_mesh = load_models()


# =========================================================
# UPLOADER
# =========================================================

st.markdown("""
<div style="
position:relative;
pointer-events:none;
z-index:1;
text-align:center;
margin-bottom:-90px;
padding:1.5rem 0 .5rem">

<div style="font-size:1.8rem;margin-bottom:.4rem">📸</div>

<div style="
color:rgba(255,255,255,.7);
font-size:.92rem;
margin-bottom:.2rem">

คลิกเพื่ออัปโหลดภาพใบหน้า

</div>

<div style="
color:rgba(255,255,255,.3);
font-size:.73rem">

JPG, JPEG, PNG · ใบหน้าเดียว · หน้าตรง · แสงสว่างเพียงพอ

</div>

</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "",
    type=["jpg","jpeg","png"],
    label_visibility="collapsed"
)


# =========================================================
# FUNCTIONS
# =========================================================

def fix_orientation(img_pil):

    try:
        from PIL import ImageOps
        return ImageOps.exif_transpose(img_pil)

    except:
        return img_pil


def get_pixel(lm, idx, ih, iw):

    pt = lm[idx]

    return (
        int(pt.x * iw),
        int(pt.y * ih)
    )


def predict_face_shape(img_pil):

    img_pil = fix_orientation(img_pil)

    img_rgb = np.array(img_pil.convert("RGB"))

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    ih, iw = img_rgb.shape[:2]

    img_out = img_rgb.copy()

    from mediapipe.tasks.python import vision as mp_vision

    img_rgb_c = np.ascontiguousarray(img_rgb.astype(np.uint8))

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb_c
    )

    results = face_mesh.detect(mp_image)

    if not results.face_landmarks:
        return None, None, None, None, img_out, False

    lm = results.face_landmarks[0]

    # face crop

    x_min = min([int(p.x * iw) for p in lm])
    x_max = max([int(p.x * iw) for p in lm])

    y_min = min([int(p.y * ih) for p in lm])
    y_max = max([int(p.y * ih) for p in lm])

    pad = 40

    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)

    x_max = min(iw, x_max + pad)
    y_max = min(ih, y_max + pad)

    face_crop = img_rgb[y_min:y_max, x_min:x_max]

    face_crop = cv2.resize(face_crop, (299,299))

    pred = face_shape_model.predict(
        np.expand_dims(face_crop, 0),
        verbose=0
    )

    probs = tf.nn.softmax(pred[0]).numpy()

    idx = np.argmax(probs)

    face_shape = classes[idx]

    confidence = probs[idx] * 100

    return face_shape, confidence, 0, 0, img_out, True


# =========================================================
# RESULT
# =========================================================

if uploaded_file:

    img_pil = Image.open(uploaded_file)

    col1, col2 = st.columns(2, gap="large")

    with col1:

        with st.spinner("🔍 กำลังวิเคราะห์..."):

            face_shape, confidence, ratiog, score, img_out, face_detected = predict_face_shape(img_pil)

        st.image(img_out, use_container_width=True)

        # ชื่อไฟล์ใต้รูป
        st.markdown(
            f"""
            <div style="
                text-align:center;
                margin-top:.7rem;
                padding:.55rem .8rem;
                background:rgba(255,255,255,.03);
                border:1px solid rgba(255,255,255,.06);
                border-radius:12px;
                color:rgba(255,255,255,.45);
                font-size:.76rem;
                overflow:hidden;
                text-overflow:ellipsis;
                white-space:nowrap;
            ">
                📄 {uploaded_file.name}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        if not face_detected:

            st.error("❌ ไม่พบใบหน้าในภาพ")

        else:

            info = shape_info[face_shape]

            st.markdown(f"""
            <div style="
                background:rgba(255,255,255,.05);
                border:1px solid rgba(255,255,255,.08);
                border-radius:24px;
                padding:1.5rem;
            ">

            <div style="font-size:2rem">{info['emoji']}</div>

            <h1 style="
                color:white;
                margin-top:.3rem;
                margin-bottom:.3rem;
            ">
                {face_shape}
            </h1>

            <p style="color:rgba(255,255,255,.5)">
                {info['desc']}
            </p>

            <h2 style="color:white;margin-top:1rem">
                {confidence:.1f}%
            </h2>

            <div style="
                background:rgba(255,255,255,.08);
                height:6px;
                border-radius:999px;
                overflow:hidden;
                margin-bottom:1rem;
            ">

            <div style="
                width:{confidence:.1f}%;
                height:100%;
                background:{info['accent']};
            ">
            </div>

            </div>

            <div style="
                background:rgba(255,255,255,.04);
                padding:1rem;
                border-radius:14px;
                margin-bottom:.7rem;
            ">
                <b>💇 ทรงผมแนะนำ</b><br><br>
                {info['hair']}
            </div>

            <div style="
                background:rgba(255,255,255,.04);
                padding:1rem;
                border-radius:14px;
            ">
                <b>👓 แว่นตาแนะนำ</b><br><br>
                {info['glasses']}
            </div>

            </div>
            """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    "<div class='footer'>Powered by <b>4 angie</b></div>",
    unsafe_allow_html=True
)
