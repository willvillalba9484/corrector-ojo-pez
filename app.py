import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import ffmpeg

st.set_page_config(page_title="Corrector Ojo de Pez Pro", layout="wide")
st.title("📹 Corrector de Video con Audio y Alta Calidad")

st.sidebar.header("🎛️ Ajustes de Lente")
k1 = st.sidebar.slider("Fuerza de Corrección (k1)", -1.0, 1.0, -0.15, 0.01)
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

uploaded_file = st.file_uploader("Sube tu video aquí", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Guardar video original en un archivo temporal
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Vista previa
    ret, frame = cap.read()
    if ret:
        st.subheader("📸 Vista previa de calibración")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado_previa = corregir_fisheye(frame_rgb, k1, zoom)
        st.image(resultado_previa, caption="Ajusta los controles antes de procesar", use_container_width=True)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if st.button("🚀 Procesar Video Completo (Alta Calidad + Audio)"):
        progreso = st.progress(0)
        status_text = st.empty()
        
        # 1. Crear video mudo temporal en ALTA CALIDAD (usando una tasa de bits alta)
        video_mudo_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_mudo_path, fourcc, fps, (width, height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_corregido = corregir_fisheye(frame, k1, zoom)
            out.write(frame_corregido)
            frame_count += 1
            porcentaje = int((frame_count / total_frames) * 100)
            progreso.progress(porcentaje)
            status_text.text(f"Procesando imágenes: {porcentaje}%")
            
        cap.release()
        out.release()
        
        # 2. Combinar el video corregido con el audio del video original usando FFmpeg
        status_text.text("🎵 Combinando pistas de audio y optimizando calidad...")
        video_final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        try:
            video_input = ffmpeg.input(video_mudo_path)
            audio_input = ffmpeg.input(tfile.name).audio
            
            # Copia el audio original y une el video con compresión de alta calidad (crf 18)
            ffmpeg.output(
                video_input, 
                audio_input, 
                video_final_path, 
                vcodec='libx264', 
                acodec='copy', 
                crf=18,
                loglevel='quiet'
            ).overwrite_output().run()
            
            status_text.success("🎉 ¡Video masterizado con éxito!")
            
            with open(video_final_path, "rb") as file:
                st.download_button(
                    label="📥 Descargar Video HD con Audio",
                    data=file,
                    file_name="video_perfecto.mp4",
                    mime="video/mp4"
                )
        except Exception as e:
            status_text.error("Hubo un problema al procesar el audio. Descarga la versión sin audio abajo.")
            with open(video_mudo_path, "rb") as file:
                st.download_button(
                    label="📥 Descargar Versión sin Audio (Alta Calidad)",
                    data=file,
                    file_name="video_mudo_hd.mp4",
                    mime="video/mp4"
                )
        
        # Limpieza
        if os.path.exists(tfile.name): os.unlink(tfile.name)
        if os.path.exists(video_mudo_path): os.unlink(video_mudo_path)
