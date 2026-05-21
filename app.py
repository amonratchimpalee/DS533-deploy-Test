import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
from PIL import Image
import gdown
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

# -----------------------------
# Register custom preprocess function (required for model deserialization)
# -----------------------------
@tf.keras.saving.register_keras_serializable()
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
def load_model():
    if not os.path.exists(MODEL_LOCAL):
        gdown.download(MODEL_URL, MODEL_LOCAL, quiet=False)
    return tf.keras.models.load_model(
        MODEL_LOCAL,
        custom_objects={'preprocess': preprocess}
    )

face_shape_model = load_model()

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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
body {background: linear-gradient(to right, #e0f7fa, #fff9c4);}
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

    # ----- MediaPipe landmark detection -----
    ratiog = 0
    score = 0
    face_detected = False

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img)

        if results.multi_face_landmarks:
            face_detected = True
            h, w = img.shape[:2]

            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=img_landmarks,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=img_landmarks,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

                lm = face_landmarks.landmark
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

    result = (
        f"Face Shape: {face_shape} ({confidence:.2f}%)\n"
        f"Hairstyle: {recommend}\n"
        f"Golden Ratio: {ratiog:.2f} | Score: {score:.2f}%"
    )
    return result, img_landmarks


if uploaded_file is not None:
    img_pil = Image.open(uploaded_file)
    result_text, landmark_img = predict_face_shape(img_pil)
    st.text_area("Prediction Result", result_text, height=120)
    if landmark_img is not None:
        st.image(landmark_img, caption="Landmarks Detected", use_column_width=True)
