
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
st.title('YOLO Object Detection')

# Инициализация модели YOLO
model = YOLO('best.pt')  # Путь к файлу с весами модели

# Загрузка изображения
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg"], key='image_uploader')
if uploaded_image is not None:
    # Преобразование изображения в формат, совместимый с моделью YOLO
    image = Image.open(uploaded_image).convert('RGB')
    image_array = np.array(image)  # Модель YOLO требует numpy array

    # Обработка изображения
    results = model(image_array)

    # Получение и отображение аннотированного изображения
    annotated_image = results.render()[0]  # Метод render возвращает изображение с аннотациями
    st.image(annotated_image, caption='Processed Image')

    # Если требуется вывести дополнительные детали предсказаний
    for pred in results.pred[0]:
        # pred - это тензор с предсказаниями [x1, y1, x2, y2, conf, class]
        class_id = int(pred[5])
        class_name = model.names[class_id]
        confidence = pred[4].item()
        st.write(f'{class_name} {confidence:.2f}')


