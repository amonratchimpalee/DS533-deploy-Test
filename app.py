import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras
import os
from PIL import Image
import gdown
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

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

# -----------------------------
# Caching model load
# -----------------------------
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_LOCAL):
        gdown.download(MODEL_URL, MODEL_LOCAL, quiet=False)
    face_model = tf.keras.models.load_model(
        MODEL_LOCAL,
        custom_objects={'preprocess': preprocess}
    )
    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_model, face_cascade

face_shape_model, face_cascade = load_models()

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

os.makedirs("saved_results", exist_ok=True)


def predict_face_shape(img_pil):
    img = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_landmarks = img.copy()

    # ----- Model prediction -----
    img_resized = cv2.resize(img_bgr, (299, 299))
    img_input = np.expand_dims(img_resized, axis=0)
    pred = face_shape_model.predict(img_input, verbose=0)
    idx = np.argmax(pred)
    face_shape = classes[idx]
    confidence = pred[0][idx] * 100

    # ----- OpenCV face detection + golden ratio -----
    ratiog = 0
    score = 0
    face_detected = False

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        face_detected = True
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(img_landmarks, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Estimate key points
            forehead_y = y
            chin_y = y + h
            left_x = x
            right_x = x + w

            face_height = float(chin_y - forehead_y)
            face_width = float(right_x - left_x)

            if face_width > 0:
                ratiog = face_height / face_width
                score = max(0.0, min((1 - abs(ratiog - 1.618) / 1.618) * 100, 100))

            # Draw key points
            cx = x + w // 2
            cv2.circle(img_landmarks, (cx, forehead_y), 5, (255, 0, 0), -1)
            cv2.circle(img_landmarks, (cx, chin_y), 5, (255, 0, 0), -1)
            cv2.circle(img_landmarks, (left_x, y + h // 2), 5, (255, 0, 0), -1)
            cv2.circle(img_landmarks, (right_x, y + h // 2), 5, (255, 0, 0), -1)
            break  # use first face only

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
        st.image(landmark_img, caption="Face Detected", use_column_width=True)
