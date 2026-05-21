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
        num_faces=10,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_mesh = mp_vision.FaceLandmarker.create_from_options(options)
    return face_model, face_mesh

classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

LANDMARK_TRICHION = 10
LANDMARK_GNATHION = 152
LANDMARK_ZY_LEFT  = 234
LANDMARK_ZY_RIGHT = 454

shape_info = {
    'Oval':   {'emoji':'🥚','color':[218,165,32],'gradient':'linear-gradient(135deg,#f7c948,#ffe08a)','accent':'#ffe08a',
               'desc':'ใบหน้ารูปไข่',
               'hair':'ผมสั้นถึงกลาง เช่น blunt bob, shoulder-length, pixie cut, long layers และหน้าม้าปัดข้าง'},
    'Square': {'emoji':'⬛','color':[210,140,0],'gradient':'linear-gradient(135deg,#d48c00,#f5c842)','accent':'#f5c842',
               'desc':'ใบหน้าเหลี่ยม',
               'hair':'ผมยาวปานกลางถึงยาว พร้อมไล่เลเยอร์หรือปลายฟุ้ง เช่น beach waves และหน้าม้านุ่มๆ'},
    'Round':  {'emoji':'⭕','color':[232,120,0],'gradient':'linear-gradient(135deg,#e87800,#ffc13b)','accent':'#ffc13b',
               'desc':'ใบหน้ากลม',
               'hair':'ทรงเพิ่มความสูงให้ใบหน้า เช่น textured bob, long layers, แสกข้าง และ blunt bangs'},
    'Heart':  {'emoji':'❤️','color':[200,150,0],'gradient':'linear-gradient(135deg,#c89600,#fada5e)','accent':'#fada5e',
               'desc':'ใบหน้ารูปหัวใจ',
               'hair':'ผมยาวระดับไหล่ พร้อมเลเยอร์บริเวณกราม curtain bangs หรือ wispy bangs'},
    'Oblong': {'emoji':'📏','color':[180,120,0],'gradient':'linear-gradient(135deg,#b47800,#f0b429)','accent':'#f0b429',
               'desc':'ใบหน้ายาว',
               'hair':'ลอนคลาย, loose curls, layered bob และหน้าม้าปัดข้างหรือ curtain bangs'},
}

st.set_page_config(page_title="Face Shape AI ✨", page_icon="✨", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── background ── */
[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]>div,.main,.block-container{background:transparent!important}
[data-testid="stAppViewContainer"]{
    background:radial-gradient(ellipse 80% 50% at 20% -10%,rgba(200,130,0,.22) 0%,transparent 60%),
               radial-gradient(ellipse 60% 40% at 80% 110%,rgba(180,80,0,.18) 0%,transparent 60%),
               #0d0a04!important;min-height:100vh}
[data-testid="stHeader"],[data-testid="stToolbar"]{background:transparent!important}
[data-testid="stDecoration"]{display:none!important}
.main .block-container{padding-top:2.5rem!important;max-width:780px}
html,body,[class*="css"],[data-testid],p,span,div,label,button{font-family:'DM Sans',sans-serif!important;color:rgba(255,255,255,.85)}

/* ── hero ── */
.hero-wrap{text-align:center;margin-bottom:1.8rem}
.hero-title{font-family:'Playfair Display',serif!important;font-size:clamp(2rem,6vw,3.2rem);font-weight:900!important;
    background:linear-gradient(135deg,#fff 0%,#ffe8a0 45%,#e8860a 100%);
    -webkit-background-clip:text!important;-webkit-text-fill-color:transparent!important;
    background-clip:text!important;letter-spacing:-.02em;line-height:1.1;margin-bottom:.4rem}
.hero-sub{color:rgba(255,255,255,.3)!important;font-size:.85rem;letter-spacing:.12em;
    text-transform:uppercase;-webkit-text-fill-color:rgba(255,255,255,.3)!important}
.divider{height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),rgba(220,150,20,.6),rgba(255,255,255,.08),transparent);margin:0 0 2rem}

/* ── file uploader ── */
[data-testid="stFileUploader"] label,[data-testid="stFileUploader"] label *{
    color:rgba(255,255,255,.7)!important;-webkit-text-fill-color:rgba(255,255,255,.7)!important;font-size:.95rem!important}
[data-testid="stFileUploader"] section{
    background:rgba(255,255,255,.04)!important;
    border:1.5px dashed rgba(255,255,255,.18)!important;
    border-radius:18px!important;
    padding:.75rem 1.2rem!important}
[data-testid="stFileUploader"] section:hover{
    border-color:rgba(220,150,20,.6)!important;
    background:rgba(220,150,20,.05)!important}
[data-testid="stFileUploaderDropzoneInstructions"],[data-testid="stFileUploaderDropzoneInstructions"] *{
    color:rgba(255,255,255,.4)!important;-webkit-text-fill-color:rgba(255,255,255,.4)!important}
/* ปุ่ม Browse files — ซ่อน icon button (position absolute ทับ text button) */
[data-testid="stFileUploader"] section{position:relative!important}
[data-testid="stFileUploader"] section button{
    background:rgba(255,255,255,.08)!important;
    border:1px solid rgba(255,255,255,.2)!important;
    border-radius:8px!important;
    color:#fff!important;
    -webkit-text-fill-color:#fff!important;
    position:relative!important;
    z-index:1!important}
[data-testid="stFileUploader"] section button:first-child{
    position:absolute!important;
    opacity:0!important;
    pointer-events:none!important;
    z-index:0!important}


/* ── image / spinner / alert ── */
[data-testid="stImage"] img{border-radius:18px!important;border:1px solid rgba(255,255,255,.1)!important;box-shadow:0 20px 60px rgba(0,0,0,.5)!important}
[data-testid="stSpinner"] *{color:rgba(255,255,255,.5)!important}
[data-testid="stAlert"]{background:rgba(233,30,99,.1)!important;border:1px solid rgba(233,30,99,.3)!important;border-radius:14px!important}
[data-testid="stAlert"] *{color:#ff6b9d!important;-webkit-text-fill-color:#ff6b9d!important}

/* ── footer ── */
.footer{text-align:center;padding:2.5rem 0 1rem;font-size:.75rem;color:rgba(255,255,255,.15)!important;
    -webkit-text-fill-color:rgba(255,255,255,.15)!important;letter-spacing:.06em}
.footer b{color:rgba(220,160,20,.6)!important;-webkit-text-fill-color:rgba(220,160,20,.6)!important}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">✨ Face Shape classification</div>
  <div class="hero-sub">วิเคราะห์รูปใบหน้าและแนะนำทรงผมด้วย </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

face_shape_model, face_mesh = load_models()
uploaded_file = st.file_uploader("📸  อัปโหลดภาพใบหน้าของคุณ", type=["jpg","jpeg","png"])
# ซ่อน icon button ตัวแรกที่ทับกับ text button ด้วย JS
st.components.v1.html("""
<script>
function hideIconBtn() {
    const btns = window.parent.document.querySelectorAll('[data-testid="stFileUploader"] section button');
    if (btns.length >= 2) {
        btns[0].style.display = 'none';
    } else {
        setTimeout(hideIconBtn, 200);
    }
}
hideIconBtn();
</script>
""", height=0)
st.markdown("""
<div style='font-size:.78rem;color:rgba(255,255,255,.3);margin-top:-.5rem;margin-bottom:1rem;line-height:1.8'>
  ℹ️ เพื่อผลลัพธ์ที่แม่นยำ: ใช้ภาพ <b style='color:rgba(255,255,255,.5)'>หน้าตรง</b> &nbsp;·&nbsp;
  แสงสว่างเพียงพอ &nbsp;·&nbsp;
&nbsp;·&nbsp;
  มองเห็นใบหน้าครบตั้งแต่หน้าผากถึงคาง
</div>
""", unsafe_allow_html=True)
os.makedirs("saved_results", exist_ok=True)


def get_pixel(lm, idx, ih, iw):
    pt = lm[idx]
    return (int(pt.x * iw), int(pt.y * ih))


def draw_landmarks_mesh(img_out, tr, gn, zy_l, zy_r, color_bgr):
    c  = color_bgr
    cw = (255, 255, 255)
    mid_x = (tr[0] + gn[0]) // 2
    cv2.line(img_out, (mid_x, tr[1]), (mid_x, gn[1]), c, 2, cv2.LINE_AA)
    cv2.line(img_out, zy_l, zy_r, c, 2, cv2.LINE_AA)
    landmarks = [
        (tr,   "A: trichion"),
        (gn,   "B: gnathion"),
        (zy_l, "C: zygion L"),
        (zy_r, "D: zygion R"),
    ]
    for pt, lbl in landmarks:
        cv2.circle(img_out, pt, 7, c,  -1, cv2.LINE_AA)
        cv2.circle(img_out, pt, 7, cw,  2, cv2.LINE_AA)
        tx = pt[0] + 10 if pt[0] < img_out.shape[1] - 80 else pt[0] - 90
        ty = pt[1] - 8  if pt[1] > 20                     else pt[1] + 18
        cv2.putText(img_out, lbl, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, cw, 1, cv2.LINE_AA)


def fix_orientation(img_pil):
    """แก้ EXIF orientation สำหรับภาพจากโทรศัพท์"""
    try:
        from PIL import ImageOps
        return ImageOps.exif_transpose(img_pil)
    except Exception:
        return img_pil

def predict_face_shape(img_pil):
    img_pil = fix_orientation(img_pil)
    img_rgb = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ih, iw  = img_rgb.shape[:2]
    img_out = img_rgb.copy()

    img_resized = cv2.resize(img_bgr, (299, 299))
    pred        = face_shape_model.predict(np.expand_dims(img_resized, 0), verbose=0)
    idx         = np.argmax(pred)
    face_shape  = classes[idx]
    confidence  = float(pred[0][idx]) * 100

    ratiog, score, face_detected = 0.0, 0.0, False

    from mediapipe.tasks.python import vision as mp_vision
    img_rgb_c = np.ascontiguousarray(img_rgb.astype(np.uint8))
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb_c)
    results   = face_mesh.detect(mp_image)

    if results.face_landmarks:
        if len(results.face_landmarks) > 1:
            return face_shape, confidence, 0.0, 0.0, img_out, "multiple"
        face_detected = True
        lm    = results.face_landmarks[0]
        c_bgr = tuple(shape_info[face_shape]['color'][::-1])

        gn   = get_pixel(lm, LANDMARK_GNATHION, ih, iw)
        zy_l = get_pixel(lm, LANDMARK_ZY_LEFT,  ih, iw)
        zy_r = get_pixel(lm, LANDMARK_ZY_RIGHT, ih, iw)

        # ── trichion: topmost landmark + offset ขึ้น ~12% ของความสูงใบหน้า ──
        all_y = [int(p.y * ih) for p in lm]
        top_y = max(0, min(all_y))
        gn_y  = gn[1]
        face_h_est = gn_y - top_y
        hairline_y = max(0, top_y - int(face_h_est * 0.12))
        mid_face_x = (zy_l[0] + zy_r[0]) // 2
        tr = (mid_face_x, hairline_y)

        gn   = (max(0, min(gn[0], iw-1)), max(0, min(gn[1], ih-1)))

        draw_landmarks_mesh(img_out, tr, gn, zy_l, zy_r, c_bgr)

        face_h_meas = abs(gn[1] - tr[1])
        face_w_meas = abs(zy_r[0] - zy_l[0])
        ratiog = face_h_meas / face_w_meas if face_w_meas > 0 else 1.0
        score  = max(0.0, min((1 - abs(ratiog - 1.618) / 1.618) * 100, 100))

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
        if face_detected == "multiple":
            st.error("❌ พบหลายใบหน้าในภาพ กรุณาอัปโหลดภาพที่มีใบหน้าเดียว")
        elif not face_detected:
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

            if 1.608 <= ratiog <= 1.628:  # ±0.01 รอบ 1.618
                fi_label = "ปกติ — Normal (≈ 1.618)"
                fi_color = "#7fff7f"
            elif ratiog > 1.628:
                fi_label = "ใบหน้ายาว — Long face (> 1.618)"
                fi_color = accent
            else:
                fi_label = "ใบหน้าสั้น — Short face (< 1.618)"
                fi_color = accent

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
.fi-val{{font-size:.9rem;font-weight:600;color:{fi_color}}}
.badge{{display:inline-block;background:rgba(255,255,255,.08);border-radius:6px;
  font-size:.6rem;color:rgba(255,255,255,.3);padding:.15rem .4rem;margin-top:.4rem;
  text-transform:uppercase;letter-spacing:.08em}}
.hair-box{{background:rgba(255,255,255,.04);border-left:3px solid {accent};
  border-radius:0 12px 12px 0;padding:.85rem 1rem}}
.hair-title{{color:{accent};font-size:.68rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem}}
.hair-text{{color:rgba(255,255,255,.68);font-size:.85rem;line-height:1.6}}
.positive-box{{margin-top:.75rem;padding:.8rem 1rem;background:rgba(255,255,255,.04);
  border-radius:13px;font-size:.8rem;color:rgba(255,255,255,.45);
  text-align:center;line-height:1.7;font-style:italic}}
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
      <div class='m-label'>Golden Ratio Score</div>
    </div>
  </div>
  <div class='fi-box'>
    <div class='fi-val'>{fi_label}</div>
  </div>
  <div class='hair-box'>
    <div class='hair-title'>💇 ทรงผมที่แนะนำ</div>
    <div class='hair-text'>{hair}</div>
  </div>
  <div class='positive-box'>
    ✨ ต่อให้ Golden Ratio Score จะเป็นเท่าไหร่ ความสวยงามไม่ได้วัดด้วยตัวเลข ทุกรูปหน้ามีเสน่ห์เฉพาะตัวที่ไม่มีใครเหมือน ✨
  </div>
</div>
</body></html>"""

            components.html(card_html, height=620, scrolling=False)

st.markdown("<div class='footer'>Powered by <b>4 angie</b></div>",
            unsafe_allow_html=True)
