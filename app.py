import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Recicjale IA-ISC", layout="centered")
st.title("Modelo predictivo Reciclaje clase de IA-ISC-Campus Comayagua-2026-Asalia Zavala")
st.write("Suba una imagen para clasificar con el modelo MobileNetV2 pre entrenado")

IMG_SIZE=(224,224)
MODEL_DIR=Path("modelo_reciclaje_mobilenet")
CLASS_PATH=MODEL_DIR/"class_names.json"
MODEL_PATHS=[MODEL_DIR/"wate_mobilenet.keras", MODEL_DIR/"waste_mobilenet.h5"]

