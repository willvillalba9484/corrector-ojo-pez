import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Corrector Ojo de Pez", layout="wide")
st.title("📹 Corrector de Video Ojo de Pez Completo")

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
    # Guardar video subido temporalmente
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Mostrar una vista previa estática para calibrar
    ret, frame = cap.read()
    if ret:
        st.subheader("📸 Vista previa de calibración")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado_previa = corregir_fisheye(frame_rgb, k1, zoom)
        st.image(resultado_previa, caption="Ajusta los controles de la izquierda antes de procesar", use_container_width=True)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reiniciar video al inicio
    
    # Botón para iniciar el procesamiento completo
    if st.button("🚀 Procesar y Convertir Video Completo"):
        progreso = st.progress(0)
        status_text = st.empty()
        
        # Archivo de salida temporal
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec compatible
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Procesar cuadro por cuadro
            frame_corregido = corregir_fisheye(frame, k1, zoom)
            out.write(frame_corregido)
            
            frame_count += 1
            porcentaje = int((frame_count / total_frames) * 100)
            progreso.progress(porcentaje)
            status_text.text(f"Procesando: {porcentaje}% completado")
            
        cap.release()
        out.release()
        
        status_text.success("🎉 ¡Video procesado con éxito!")
        
        # Botón para descargar el resultado al celular
        with open(output_path, "rb") as file:
            st.download_button(
                label="📥 Descargar Video Corregido",
                data=file,
                file_name="video_sin_ojo_de_pez.mp4",
                mime="video/mp4"
            )
            
        # Limpieza de temporales
        os.unlink(tfile.name)
