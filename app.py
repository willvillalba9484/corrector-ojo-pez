import streamlit as st
import cv2
import numpy as np
import tempfile

st.set_page_config(page_title="Corrector Ojo de Pez", layout="wide")
st.title("📹 Corrector de Ojo de Pez")

st.sidebar.header("🎛️ Ajustes")
k1 = st.sidebar.slider("Corrección (k1)", -1.0, 1.0, -0.15, 0.01)
zoom = st.sidebar.slider("Zoom", 0.3, 2.0, 0.8, 0.05)

def corregir_fisheye(frame, k1, zoom_val):
    h, w = frame.shape[:2]
    K = np.array([[w * 0.6, 0, w / 2], [0, h * 0.6, h / 2], [0, 0, 1]])
    D = np.array([[k1, 0.0, 0.0, 0.0]])
    scaled_K = K.copy()
    scaled_K[0, 0] *= zoom_val
    scaled_K[1, 1] *= zoom_val
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), scaled_K, (w, h), cv2.CV_16SC2)
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

uploaded_file = st.file_uploader("Sube tu video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = corregir_fisheye(frame_rgb, k1, zoom)
        st.image(resultado, use_container_width=True)
    cap.release()
