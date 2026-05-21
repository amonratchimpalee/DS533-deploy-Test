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
  <div class="hero-sub">วิเคราะห์รูปใบหน้าและแนะนำทรงผมด้วย AI</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

face_shape_model, face_cascade = load_models()
uploaded_file = st.file_uploader("📸  อัปโหลดภาพใบหน้าของคุณ", type=["jpg","jpeg","png"])
os.makedirs("saved_results", exist_ok=True)


def find_landmarks(img_rgb, face_rect):
    """
    หา landmark 4 จุดตาม Saraswathi (2007) / PMC3530317:
      tr  = trichion  → hairline (topmost skin pixel บน midline)
      gn  = gnathion  → lower border of chin (bottommost skin pixel)
      zy_l/zy_r = zygion → prominent lateral points of zygomatic bones
                           (~40-45% ของ face height จาก hairline ลงมา)

    Facial Index = vertical length (tr→gn) / bizygomatic width (zy_l→zy_r)
    Normal = 1.6, Long > 1.6, Short < 1.6
    """
    ih, iw = img_rgb.shape[:2]
    x, y, w, h = face_rect

    # ── ขยาย ROI ให้ครอบ hairline จริง (tr) และปลายคาง (gn) ──
    # Haar cascade มักตัด hairline ออก จึงต้อง pad_top มากพอ
    pad_top    = int(h * 0.90)   # ~90% เพื่อให้ถึง hairline จริง
    pad_bottom = int(h * 0.30)   # ~30% เพื่อให้ถึงปลายคาง
    pad_side   = int(w * 0.15)

    roi_y1 = max(0, y - pad_top)
    roi_y2 = min(ih, y + h + pad_bottom)
    roi_x1 = max(0, x - pad_side)
    roi_x2 = min(iw, x + w + pad_side)

    crop = img_rgb[roi_y1:roi_y2, roi_x1:roi_x2]

    # ── Skin mask ใน YCrCb (range มาตรฐาน) ──
    ycrcb     = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb)
    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    kernel    = np.ones((11, 11), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE,  kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── fallback: ถ้าหา skin contour ไม่ได้ ใช้ ROI โดยตรง ──
    if not contours:
        cx   = (roi_x1 + roi_x2) // 2
        tr   = (cx, roi_y1)
        gn   = (cx, roi_y2)
        zy_l = (roi_x1, y + int(h * 0.42))
        zy_r = (roi_x2, y + int(h * 0.42))
        return tr, gn, zy_l, zy_r, roi_y2 - roi_y1, roi_x2 - roi_x1

    largest = max(contours, key=cv2.contourArea)
    pts     = largest[:, 0, :]  # (N, 2)  col-0=x, col-1=y  (relative to crop)

    # ── trichion (tr): topmost skin pixel บน midline ──
    # ใช้ topmost point ของ contour (y น้อยสุด)
    top_y   = int(np.min(pts[:, 1]))
    # x ของ topmost: ค่า x เฉลี่ยของ points ที่อยู่ใน top 5% เพื่อให้ใกล้ midline
    top_band = pts[pts[:, 1] <= top_y + max(int((np.max(pts[:,1])-top_y)*0.05), 3)]
    top_x    = int(np.mean(top_band[:, 0]))

    # ── gnathion (gn): bottommost skin pixel บน midline ──
    bot_y   = int(np.max(pts[:, 1]))
    bot_band = pts[pts[:, 1] >= bot_y - max(int((bot_y-top_y)*0.05), 3)]
    bot_x    = int(np.mean(bot_band[:, 0]))

    tr_abs = (roi_x1 + top_x, roi_y1 + top_y)
    gn_abs = (roi_x1 + bot_x, roi_y1 + bot_y)

    # ── zygion (zy): prominent lateral points ของ zygomatic bones ──
    # ตาม paper: "prominent points laterally on body of both zygomatic bones"
    # ≈ 40–45% ของ face height ลงมาจาก hairline (ระดับโหนกแก้ม)
    face_h_px = gn_abs[1] - tr_abs[1]
    zy_y_abs  = tr_abs[1] + int(face_h_px * 0.42)

    # หา leftmost / rightmost ของ skin contour ในแนว horizontal band ที่ zy_y
    tol      = max(int(face_h_px * 0.08), 5)
    zy_y_rel = zy_y_abs - roi_y1
    zy_band  = pts[(pts[:, 1] >= zy_y_rel - tol) & (pts[:, 1] <= zy_y_rel + tol)]

    if len(zy_band) >= 2:
        left_x  = roi_x1 + int(np.min(zy_band[:, 0]))
        right_x = roi_x1 + int(np.max(zy_band[:, 0]))
    else:
        # fallback: leftmost/rightmost ทั้ง contour
        left_x  = roi_x1 + int(np.min(pts[:, 0]))
        right_x = roi_x1 + int(np.max(pts[:, 0]))

    zy_l = (left_x,  zy_y_abs)
    zy_r = (right_x, zy_y_abs)

    face_w_meas = right_x - left_x

    return tr_abs, gn_abs, zy_l, zy_r, face_h_px, face_w_meas


def draw_landmarks(img_out, tr, gn, zy_l, zy_r, color_bgr):
    """วาดเส้นวัดและ landmark ตาม Saraswathi (2007) Fig.1"""
    c  = color_bgr
    cw = (255, 255, 255)

    # เส้น AB = vertical length (tr→gn) — midline
    mid_x = (tr[0] + gn[0]) // 2
    cv2.line(img_out, (mid_x, tr[1]), (mid_x, gn[1]), c,  2, cv2.LINE_AA)

    # เส้น CD = bizygomatic width (zy_l→zy_r)
    cv2.line(img_out, zy_l, zy_r, c, 2, cv2.LINE_AA)

    # landmark dots + labels
    landmarks = [(tr, "A (tr)"), (gn, "B (gn)"), (zy_l, "C (zy)"), (zy_r, "D (zy)")]
    for pt, lbl in landmarks:
        cv2.circle(img_out, pt, 7, c,   -1,            cv2.LINE_AA)
        cv2.circle(img_out, pt, 7, cw,   2,            cv2.LINE_AA)
        # offset label เพื่อไม่ทับจุด
        tx = pt[0] + 10 if pt[0] < img_out.shape[1] - 60 else pt[0] - 70
        ty = pt[1] - 8  if pt[1] > 20                     else pt[1] + 18
        cv2.putText(img_out, lbl, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cw, 1, cv2.LINE_AA)


def predict_face_shape(img_pil):
    img     = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_out = img.copy()

    # ── ทำนาย face shape ด้วยโมเดล ──
    img_resized = cv2.resize(img_bgr, (299, 299))
    pred        = face_shape_model.predict(np.expand_dims(img_resized, 0), verbose=0)
    idx         = np.argmax(pred)
    face_shape  = classes[idx]
    confidence  = float(pred[0][idx]) * 100

    ratiog, score, face_detected = 0.0, 0.0, False

    # ── ตรวจจับใบหน้าด้วย Haar Cascade ──
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    if len(faces) > 0:
        face_detected = True
        x, y, w, h = faces[0]
        c_bgr = tuple(shape_info[face_shape]['color'][::-1])

        # วาด Haar bounding box (อ้างอิง)
        cv2.rectangle(img_out, (x, y), (x+w, y+h), c_bgr, 1)

        # ── หา landmark ที่ถูกต้องตามงานวิจัย ──
        tr, gn, zy_l, zy_r, face_h_meas, face_w_meas = find_landmarks(img, (x, y, w, h))

        # ── วาด landmark + เส้นวัด ──
        draw_landmarks(img_out, tr, gn, zy_l, zy_r, c_bgr)

        # ── Facial Index = vertical length / bizygomatic width (Saraswathi 2007) ──
        if face_w_meas > 0:
            ratiog = face_h_meas / face_w_meas
        else:
            ratiog = 1.0

        # score: ใกล้ 1.6 มากแค่ไหน (0–100%)
        score = max(0.0, min((1 - abs(ratiog - 1.6) / 1.6) * 100, 100))

        # ── label Facial Index บนภาพ ──
        label = f"Facial Index: {ratiog:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx, ly = x, max(y - 12, th + 4)
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

            # face shape label ตาม Saraswathi 2007
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
    <div class='fi-label'>ผลตาม Saraswathi (2007) · Length/Bizygomatic Width</div>
    <div class='fi-val'>{fi_label}</div>
  </div>
  <div class='hair-box'>
    <div class='hair-title'>💇 ทรงผมที่แนะนำ</div>
    <div class='hair-text'>{hair}</div>
  </div>
</div>
</body></html>"""

            components.html(card_html, height=510, scrolling=False)

st.markdown("<div class='footer'>Powered by <b>4 angie</b> · อ้างอิง: Saraswathi (2007) Eur J Anat 11(3):177-180</div>",
            unsafe_allow_html=True)
