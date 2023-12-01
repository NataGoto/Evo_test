
import os
import sys
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import json
from pytube import YouTube
import tempfile
import pandas as pd
import shutil
import numpy as np 

# Добавляем текущую директорию в путь поиска модулей
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_directory)

# Streamlit интерфейс
st.title('Evodrone Test')

# Load a pretrained YOLOv8n model
model = YOLO('best.pt')

# Загрузка изображения
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg"], key='image_uploader')
if uploaded_image is not None:
    # Сохраняем загруженное изображение во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_image.read())
        tmp_file_path = tmp_file.name

    # Выполнение предсказания
    results = model(tmp_file_path)

    # Получение аннотированного изображения
    annotated_image = Image.fromarray(np.squeeze(results.render()))

    # Отображение обработанного изображения
    st.image(annotated_image, caption='Processed Image')

