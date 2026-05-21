import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras
import mediapipe as mp
import os
from PIL import Image
import gdown
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# -----------------------------
# Register custom preprocess function
# -----------------------------
@keras.saving.register_keras_serializable()
def preprocess(x):
    x = tf.cast(x, tf.float32)
    return preprocess_input(x)

# -----------------------------
# Google Drive Model URL
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1p3veX7I7_6WBM97jOSfQpSGcxwIuijD1"
MODEL_LOCAL = "best_inceptionresnetv2_face_shape_fixed.keras"

# MediaPipe face landmarker model
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_LANDMARKER_LOCAL = "face_landmarker.task"

# -----------------------------
# Caching model load
# -----------------------------
@st.cache_resource
def load_models():
    # Load face shape model
    if not os.path.exists(MODEL_LOCAL):
        gdown.download(MODEL_URL, MODEL_LOCAL, quiet=False)
    face_model = tf.keras.models.load_model(
        MODEL_LOCAL,
        custom_objects={'preprocess': preprocess}
    )

    # Download mediapipe task model
    if not os.path.exists(FACE_LANDMARKER_LOCAL):
        import urllib.request
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_LOCAL)

    return face_model

face_shape_model = load_models()

# -----------------------------
# Class labels & hairstyle recommendations
# -----------------------------
classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
hairstyle_recommendations = {
    'Oval': 'ผมสั้นถึงกลาง เช่น blunt bob, shoulder-length, pixie cut, long layers และหน้าม้าปัดข้าง',
    'Square': 'ผมยาวปานกลางถึงยาว พร้อมไล่เลเยอร์หรือปลายฟุ้ง เช่น beach waves และหน้าม้านุ่มๆ',
    'Round': 'ทรงเพิ่มความสูงให้ใบหน้า เช่น textured bob, long layers, แสกข้าง และ blunt bangs',
    'Heart': 'ผมยาวระดับไหล่ พร้อมเลเยอร์บริเวณกราม curtain bangs หรือ wispy bangs',
    'Oblong': 'ลอนคลาย, loose curls, layered bob และหน้าม้าปัดข้างหรือ curtain bangs'
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Face Shape Detector", layout="centered")
st.markdown("""
<style>
h1 {text-align:center; color:#004d40;}
.stButton>button {background-color:#00796b; color:white; border-radius:8px; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

st.title("Face Shape Detector & Hairstyle Recommendation")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

SAVE_DIR = "saved_results"
os.makedirs(SAVE_DIR, exist_ok=True)


def predict_face_shape(img_pil):
    img = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_landmarks = img.copy()

    # ----- Model prediction -----
    img_resized = cv2.resize(img_bgr, (299, 299))
    img_input = np.expand_dims(img_resized, axis=0)
    pred = face_shape_model.predict(img_input, verbose=0)
    idx = np.argmax(pred)
    face_shape = classes[idx]
    confidence = pred[0][idx] * 100

    # ----- MediaPipe new API landmark detection -----
    ratiog = 0
    score = 0
    face_detected = False

    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_LOCAL),
        running_mode=RunningMode.IMAGE,
        num_faces=1
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            face_detected = True
            h, w = img.shape[:2]
            lm = result.face_landmarks[0]

            # Draw landmarks
            for landmark in lm:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(img_landmarks, (x, y), 1, (0, 255, 0), -1)

            # Golden ratio
            forehead_y = lm[10].y * h
            chin_y = lm[152].y * h
            left_x = lm[234].x * w
            right_x = lm[454].x * w

            face_height = abs(chin_y - forehead_y)
            face_width = abs(right_x - left_x)

            if face_width > 0:
                ratiog = face_height / face_width
                score = max(0, min((1 - abs(ratiog - 1.618) / 1.618) * 100, 100))

    if not face_detected:
        return "No face detected", None

    recommend = hairstyle_recommendations[face_shape]
    result_text = (
        f"Face Shape: {face_shape} ({confidence:.2f}%)\n"
        f"Hairstyle: {recommend}\n"
        f"Golden Ratio: {ratiog:.2f} | Score: {score:.2f}%"
    )
    return result_text, img_landmarks


if uploaded_file is not None:
    img_pil = Image.open(uploaded_file)
    result_text, landmark_img = predict_face_shape(img_pil)
    st.text_area("Prediction Result", result_text, height=120)
    if landmark_img is not None:
        st.image(landmark_img, caption="Landmarks Detected", use_column_width=True)
