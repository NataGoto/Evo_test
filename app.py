import os
import sys
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import json
from pytube import YouTube
import tempfile
import time
import pandas as pd
import shutil
import subprocess 
# Добавляем текущую директорию в путь поиска модулей
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_directory)



# Функция для проверки свободного места на диске
def check_disk_space(directory, max_size_mb):
    total_size = 0
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
    return (total_size / (1024 * 1024)) <= max_size_mb

# Функция для обработки и сохранения изображения или видео
def process_and_save(uploaded_file, save_directory):
    # Путь к временному файлу
    temp_output_path = tempfile.mktemp(suffix=f'.{uploaded_file.type.split("/")[-1]}')
    
    # Запись содержимого загруженного файла во временный файл
    with open(temp_output_path, 'wb') as file:
        file.write(uploaded_file.getvalue())

    # Обработка файла моделью YOLO
    model = YOLO('best.pt')
    results = model(temp_output_path, stream=True)

        
# Streamlit интерфейс
st.title('Evodrone Test')

# Загрузка файла
uploaded_file = st.file_uploader("Upload a file", type=["png", "jpg", "mp4", "avi"])

# После загрузки файла, обрабатываем его
if uploaded_file is not None:
    # Проверяем, достаточно ли места для сохранения файла
    if check_disk_space(SAVE_DIRECTORY, max_size_mb=512):
        processed_file_path = process_and_save(uploaded_file, SAVE_DIRECTORY)
        
        # Отображаем обработанный файл в зависимости от его типа
        if uploaded_file.type in ["image/png", "image/jpeg"]:
            st.image(processed_file_path, caption='Processed Image')
        elif uploaded_file.type in ["video/mp4", "video/avi"]:
            st.video(processed_video_path)
    else:
        st.error("Недостаточно места для сохранения файла.")


youtube_url = st.text_input('Enter a YouTube URL', key='youtube_url_input')
if youtube_url:
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension='mp4').first()
    if stream:
        # Скачиваем видео
        youtube_video_path = stream.download(output_path=tempfile.gettempdir())

        # Проверяем наличие свободного места
        if check_disk_space(SAVE_DIRECTORY, max_size_mb=512):
            # Обрабатываем скачанное видео с помощью YOLO
            model = YOLO('best.pt')
            results = model(youtube_video_path, stream=True)

            # Сохраняем обработанное видео в указанную директорию
            final_output_path = os.path.join(SAVE_DIRECTORY, os.path.basename(youtube_video_path))
            shutil.move(youtube_video_path, final_output_path)
            st.video(final_output_path)
        else:
            st.error("Недостаточно места для сохранения файла.")
    else:
        st.error("Не удалось найти подходящий поток видео.")




# Загрузка и обработка видео с YouTube
youtube_url = st.text_input('Enter a YouTube URL')
if youtube_url:
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension='mp4').first()
    youtube_video = stream.download()
    if check_disk_space(SAVE_DIRECTORY, max_size_mb=512):
        processed_youtube_video_path = process_and_save(youtube_video, SAVE_DIRECTORY)
        st.video(processed_youtube_video_path)
    else:
        st.error("Недостаточно места для сохранения файла.")



# Функция для преобразования результатов в JSON, CSV, текст
def convert_results_to_formats(results):
    df = results.pandas().xyxy[0]  # Преобразование результатов YOLO в DataFrame
    json_result = df.to_json(orient="records")
    csv_result = df.to_csv(index=False)
    text_result = df.to_string()
    return json_result, csv_result, text_result

# Streamlit интерфейс для скачивания результатов
if uploaded_file is not None:
    # Обработка файла
    if check_disk_space(SAVE_DIRECTORY, max_size_mb=512):
        processed_file_path, results = process_and_save(uploaded_file, SAVE_DIRECTORY)
        
        # Отображение обработанного файла
        if uploaded_file.type.split('/')[0] == 'image':
            st.image(processed_file_path, caption='Processed Image')
        elif uploaded_file.type.split('/')[0] == 'video':
            st.video(processed_file_path)

        # Преобразование результатов и создание кнопок для скачивания
        if results:
            json_result, csv_result, text_result = convert_results_to_formats(results)
            st.download_button('Download JSON', json_result, file_name='results.json')
            st.download_button('Download CSV', csv_result, file_name='results.csv')
            st.download_button('Download Text', text_result, file_name='results.txt')
    else:
        st.error("Недостаточно места для сохранения файла.")

remove_old_files(PROCESSED_VIDEOS_PATH)
git_commit_changes()
