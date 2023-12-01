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

# Загрузка изображения
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg"], key='image_uploader')

# Загрузка видео
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"], key='video_uploader')

# Ввод YouTube URL
youtube_url = st.text_input('Enter a YouTube URL', key='youtube_url_input')

# Функция для преобразования результатов в JSON, CSV, текст
def convert_results_to_formats(results):
    df = pd.DataFrame(results)  # Преобразование списка словарей в DataFrame
    json_result = df.to_json(orient="records")
    csv_result = df.to_csv(index=False)
    text_result = df.to_string()
    return json_result, csv_result, text_result

# Обработка загруженного изображения
if uploaded_image is not None:
    model = YOLO('best.pt')
    results = model(uploaded_image, stream=True)

    # Отображение обработанного изображения
    st.image(uploaded_image, caption='Processed Image')

    # Преобразование результатов и создание кнопок для скачивания
    if results:
        json_result, csv_result, text_result = convert_results_to_formats(results)
        st.download_button('Download JSON', json_result, file_name='results.json')
        st.download_button('Download CSV', csv_result, file_name='results.csv')
        st.download_button('Download Text', text_result, file_name='results.txt')

# Обработка загруженного видео
if uploaded_video is not None:
    model = YOLO('best.pt')
    results = model(uploaded_video, stream=True)

    # Отображение обработанного видео
    st.video(uploaded_video)

    # Преобразование результатов и создание кнопок для скачивания
    if results:
        json_result, csv_result, text_result = convert_results_to_formats(results)
        st.download_button('Download JSON', json_result, file_name='results.json')
        st.download_button('Download CSV', csv_result, file_name='results.csv')
        st.download_button('Download Text', text_result, file_name='results.txt')

# Обработка YouTube видео
if youtube_url:
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension='mp4').first()
    if stream:
        # Скачиваем видео
        youtube_video_path = stream.download(output_path=tempfile.gettempdir())
        model = YOLO('best.pt')
        results = model(youtube_video_path, stream=True)

        # Показываем результаты предикта
        st.write(results)

# Кнопка для очистки кэша
if st.button('Очистить кэш'):
    st.caching.clear_cache()

