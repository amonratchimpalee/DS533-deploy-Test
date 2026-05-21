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

    # MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    return face_model, face_mesh

classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# MediaPipe Face Mesh landmark indices (based on canonical 468-point map)
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LANDMARK_TRICHION  = 10    # hairline / top of forehead
LANDMARK_GNATHION  = 152   # bottom of chin
LANDMARK_ZY_LEFT   = 234   # left zygion (cheekbone)
LANDMARK_ZY_RIGHT  = 454   # right zygion (cheekbone)
LANDMARK_GO_LEFT   = 172   # left gonion (jaw angle)  — bonus for jaw width
LANDMARK_GO_RIGHT  = 397   # right gonion (jaw angle)

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

st.set_page_config(page_title="Face Shape AI ✨", page_icon="✨", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]>div,.main,.block-container{background:transparent!important}
[data-testid="stAppViewContainer"]{
    background:radial-gradient(ellipse 80% 50% at 20% -10%,rgba(200,130,0,.22) 0%,transparent 60%),
               radial-gradient(ellipse 60% 40% at 80% 110%,rgba(180,80,0,.18) 0%,transparent 60%),
               #0d0a04!important;min-height:100vh}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stToolbar"]{background:transparent!important}
[data-testid="stDecoration"]{display:none!important}
.main .block-container{padding-top:2.5rem!important;max-width:780px}
html,body,[class*="css"],[data-testid],p,span,div,label,button{font-family:'DM Sans',sans-serif!important;color:rgba(255,255,255,.85)}
.hero-wrap{text-align:center;margin-bottom:1.8rem}
.hero-title{font-family:'Playfair Display',serif!important;font-size:clamp(2rem,6vw,3.2rem);font-weight:900!important;
    background:linear-gradient(135deg,#fff 0%,#ffe8a0 45%,#e8860a 100%);
    -webkit-background-clip:text!important;-webkit-text-fill-color:transparent!important;
    background-clip:text!important;letter-spacing:-.02em;line-height:1.1;margin-bottom:.4rem}
.hero-sub{color:rgba(255,255,255,.3)!important;font-size:.85rem;letter-spacing:.12em;
    text-transform:uppercase;-webkit-text-fill-color:rgba(255,255,255,.3)!important}
.divider{height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),rgba(220,150,20,.6),rgba(255,255,255,.08),transparent);margin:0 0 2rem}
[data-testid="stFileUploaderDropzone"],[data-testid="stFileUploader"] section{
    background:rgba(255,255,255,.04)!important;border:1.5px dashed rgba(255,255,255,.18)!important;border-radius:18px!important}
[data-testid="stFileUploaderDropzone"]:hover,[data-testid="stFileUploader"] section:hover{
    border-color:rgba(220,150,20,.6)!important;background:rgba(220,150,20,.05)!important}
[data-testid="stFileUploaderDropzoneInstructions"],[data-testid="stFileUploaderDropzoneInstructions"] *{
    color:rgba(255,255,255,.4)!important;-webkit-text-fill-color:rgba(255,255,255,.4)!important}
[data-testid="stFileUploader"] label,[data-testid="stFileUploader"] label *{
    color:rgba(255,255,255,.7)!important;-webkit-text-fill-color:rgba(255,255,255,.7)!important;font-size:.95rem!important}
[data-testid="stFileUploader"] button{background:rgba(255,255,255,.08)!important;border:1px solid rgba(255,255,255,.2)!important;
    border-radius:8px!important;color:#fff!important;-webkit-text-fill-color:#fff!important}
[data-testid="stImage"] img{border-radius:18px!important;border:1px solid rgba(255,255,255,.1)!important;box-shadow:0 20px 60px rgba(0,0,0,.5)!important}
[data-testid="stSpinner"] *{color:rgba(255,255,255,.5)!important}
[data-testid="stAlert"]{background:rgba(233,30,99,.1)!important;border:1px solid rgba(233,30,99,.3)!important;border-radius:14px!important}
[data-testid="stAlert"] *{color:#ff6b9d!important;-webkit-text-fill-color:#ff6b9d!important}
.footer{text-align:center;padding:2.5rem 0 1rem;font-size:.75rem;color:rgba(255,255,255,.15)!important;
    -webkit-text-fill-color:rgba(255,255,255,.15)!important;letter-spacing:.06em}
.footer b{color:rgba(220,160,20,.6)!important;-webkit-text-fill-color:rgba(220,160,20,.6)!important}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">✨ Face Shape AI</div>
  <div class="hero-sub">วิเคราะห์รูปใบหน้าและแนะนำทรงผมด้วย AI · MediaPipe Face Mesh</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

face_shape_model, face_mesh = load_models()
uploaded_file = st.file_uploader("📸  อัปโหลดภาพใบหน้าของคุณ", type=["jpg","jpeg","png"])
os.makedirs("saved_results", exist_ok=True)


def get_pixel(lm, idx, ih, iw):
    """แปลง normalized landmark → pixel coordinates"""
    pt = lm[idx]
    return (int(pt.x * iw), int(pt.y * ih))


def draw_landmarks_mesh(img_out, tr, gn, zy_l, zy_r, color_bgr):
    """วาดเส้นวัดและ landmark แม่นยำจาก MediaPipe"""
    c  = color_bgr
    cw = (255, 255, 255)

    # เส้น AB = vertical length (tr→gn)
    mid_x = (tr[0] + gn[0]) // 2
    cv2.line(img_out, (mid_x, tr[1]), (mid_x, gn[1]), c, 2, cv2.LINE_AA)

    # เส้น CD = bizygomatic width (zy_l→zy_r)
    cv2.line(img_out, zy_l, zy_r, c, 2, cv2.LINE_AA)

    # landmark dots + labels
    landmarks = [
        (tr,   "A · trichion"),
        (gn,   "B · gnathion"),
        (zy_l, "C · zygion"),
        (zy_r, "D · zygion"),
    ]
    for pt, lbl in landmarks:
        cv2.circle(img_out, pt, 7, c,  -1, cv2.LINE_AA)
        cv2.circle(img_out, pt, 7, cw,  2, cv2.LINE_AA)
        tx = pt[0] + 10 if pt[0] < img_out.shape[1] - 80 else pt[0] - 90
        ty = pt[1] - 8  if pt[1] > 20                     else pt[1] + 18
        cv2.putText(img_out, lbl, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, cw, 1, cv2.LINE_AA)


def predict_face_shape(img_pil):
    img_rgb = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ih, iw  = img_rgb.shape[:2]
    img_out = img_rgb.copy()

    # ── ทำนาย face shape ด้วยโมเดล ──
    img_resized = cv2.resize(img_bgr, (299, 299))
    pred        = face_shape_model.predict(np.expand_dims(img_resized, 0), verbose=0)
    idx         = np.argmax(pred)
    face_shape  = classes[idx]
    confidence  = float(pred[0][idx]) * 100

    ratiog, score, face_detected = 0.0, 0.0, False

    # ── MediaPipe Face Mesh ──
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        face_detected = True
        lm = results.multi_face_landmarks[0].landmark
        c_bgr = tuple(shape_info[face_shape]['color'][::-1])

        # ── ดึง landmark 4 จุดตาม Saraswathi (2007) ──
        tr   = get_pixel(lm, LANDMARK_TRICHION,  ih, iw)   # hairline (top)
        gn   = get_pixel(lm, LANDMARK_GNATHION,  ih, iw)   # chin bottom
        zy_l = get_pixel(lm, LANDMARK_ZY_LEFT,   ih, iw)   # left cheekbone
        zy_r = get_pixel(lm, LANDMARK_ZY_RIGHT,  ih, iw)   # right cheekbone

        # ── วาด landmark + เส้นวัด ──
        draw_landmarks_mesh(img_out, tr, gn, zy_l, zy_r, c_bgr)

        # ── Facial Index = vertical length / bizygomatic width ──
        face_h_meas = abs(gn[1] - tr[1])
        face_w_meas = abs(zy_r[0] - zy_l[0])

        if face_w_meas > 0:
            ratiog = face_h_meas / face_w_meas
        else:
            ratiog = 1.0

        score = max(0.0, min((1 - abs(ratiog - 1.6) / 1.6) * 100, 100))

        # ── label Facial Index บนภาพ ──
        label = f"Facial Index: {ratiog:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx = min(zy_l[0], tr[0])
        ly = max(tr[1] - 12, th + 4)
        cv2.rectangle(img_out, (lx-2, ly-th-4), (lx+tw+4, ly+4), (0,0,0), -1)
        cv2.putText(img_out, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_bgr, 1, cv2.LINE_AA)

    return face_shape, confidence, ratiog, score, img_out, face_detected


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

            card_html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<link href='https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap' rel='stylesheet'>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:'DM Sans',sans-serif}}
.card{{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);
  border-radius:24px;padding:1.5rem;position:relative;overflow:hidden}}
.card::before{{content:'';position:absolute;inset:0;border-radius:24px;padding:1.5px;
  background:{gradient};
  -webkit-mask:linear-gradient(#fff 0 0) content-box,linear-gradient(#fff 0 0);
  -webkit-mask-composite:xor;mask-composite:exclude;opacity:.7;pointer-events:none}}
.emoji{{font-size:2.2rem;margin-bottom:.3rem}}
.shape-name{{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:900;
  color:#fff;line-height:1.1;margin-bottom:.3rem}}
.desc{{color:rgba(255,255,255,.4);font-size:.83rem;line-height:1.6;margin-bottom:.9rem}}
.conf-label{{font-size:.65rem;color:rgba(255,255,255,.28);text-transform:uppercase;
  letter-spacing:.09em;margin-bottom:.2rem}}
.conf-value{{font-size:1.8rem;font-weight:700;color:#fff;margin-bottom:.35rem}}
.bar-bg{{background:rgba(255,255,255,.08);border-radius:99px;height:5px;overflow:hidden;margin-bottom:1rem}}
.bar-fill{{height:100%;border-radius:99px;background:{gradient};width:{conf_str}%}}
.metrics{{display:flex;gap:.55rem;margin-bottom:.75rem}}
.metric{{flex:1;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);
  border-radius:13px;padding:.8rem .4rem;text-align:center}}
.m-icon{{font-size:.95rem;margin-bottom:.12rem}}
.m-val{{font-size:1.3rem;font-weight:700;color:#fff;line-height:1}}
.m-label{{font-size:.62rem;color:rgba(255,255,255,.28);text-transform:uppercase;
  letter-spacing:.06em;margin-top:.18rem}}
.fi-box{{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
  border-radius:13px;padding:.7rem 1rem;margin-bottom:.75rem;text-align:center}}
.fi-label{{font-size:.62rem;color:rgba(255,255,255,.28);text-transform:uppercase;
  letter-spacing:.08em;margin-bottom:.2rem}}
.fi-val{{font-size:.9rem;font-weight:600;color:{accent}}}
.badge{{display:inline-block;background:rgba(255,255,255,.08);border-radius:6px;
  font-size:.6rem;color:rgba(255,255,255,.3);padding:.15rem .4rem;margin-bottom:.5rem;
  text-transform:uppercase;letter-spacing:.08em}}
.hair-box{{background:rgba(255,255,255,.04);border-left:3px solid {accent};
  border-radius:0 12px 12px 0;padding:.85rem 1rem}}
.hair-title{{color:{accent};font-size:.68rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem}}
.hair-text{{color:rgba(255,255,255,.68);font-size:.85rem;line-height:1.6}}
</style></head><body>
<div class='card'>
  <div class='emoji'>{emoji}</div>
  <div class='shape-name'>{face_shape}</div>
  <div class='desc'>{desc}</div>
  <div class='conf-label'>ความมั่นใจของโมเดล</div>
  <div class='conf-value'>{conf_str}%</div>
  <div class='bar-bg'><div class='bar-fill'></div></div>
  <div class='metrics'>
    <div class='metric'>
      <div class='m-icon'>📐</div>
      <div class='m-val'>{ratio_str}</div>
      <div class='m-label'>Facial Index</div>
    </div>
    <div class='metric'>
      <div class='m-icon'>⭐</div>
      <div class='m-val'>{score_str}%</div>
      <div class='m-label'>ใกล้ phi (1.6)</div>
    </div>
  </div>
  <div class='fi-box'>
    <div class='fi-label'>ผลตาม Saraswathi (2007) · Length / Bizygomatic Width</div>
    <div class='fi-val'>{fi_label}</div>
    <div class='badge' style='margin-top:.4rem'>📍 MediaPipe Face Mesh 468-point</div>
  </div>
  <div class='hair-box'>
    <div class='hair-title'>💇 ทรงผมที่แนะนำ</div>
    <div class='hair-text'>{hair}</div>
  </div>
</div>
</body></html>"""

            components.html(card_html, height=520, scrolling=False)

st.markdown("<div class='footer'>Powered by <b>4 angie</b> · MediaPipe Face Mesh · อ้างอิง: Saraswathi (2007) Eur J Anat 11(3):177-180</div>",
            unsafe_allow_html=True)
