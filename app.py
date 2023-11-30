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

# Добавляем текущую директорию в путь поиска модулей
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_directory)

# Путь к папке для сохранения обработанных файлов
SAVE_DIRECTORY = 'processed_videos'

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
    # Проверяем, достаточно ли места для сохранения файла
    if not check_disk_space(save_directory, max_size_mb=512):
        print("Недостаточно места на диске для сохранения файла")
        return None

    # Определяем путь к временному файлу
    temp_output_path = tempfile.mktemp(suffix=f'.{uploaded_file.type.split("/")[-1]}')
    
    # Записываем содержимое загруженного файла во временный файл
    with open(temp_output_path, 'wb') as file:
        file.write(uploaded_file.getvalue())

    # Обработка файла моделью YOLO
    model = YOLO('best.pt')  # Убедитесь, что у вас установлена модель
    results = model(temp_output_path, stream=True)

    # Перемещаем файл в директорию репозитория
    final_output_path = os.path.join(save_directory, uploaded_file.name)
    shutil.move(temp_output_path, final_output_path)
    return final_output_path

# Streamlit интерфейс
st.title('Evodrone Test')

# Загрузка и обработка изображения
image_file = st.file_uploader('Upload an image', type=['png', 'jpg'])
if image_file is not None:
    if check_disk_space(SAVE_DIRECTORY, max_size_mb=512):
        processed_image_path = process_and_save(image_file, SAVE_DIRECTORY)
        st.image(processed_image_path, caption='Processed Image')
    else:
        st.error("Недостаточно места для сохранения файла.")

# Загрузка и обработка видео
video_file = st.file_uploader('Upload a video', type=['mp4'])
if video_file is not None:
    if check_disk_space(SAVE_DIRECTORY, max_size_mb=512):
        processed_video_path = process_and_save(video_file, SAVE_DIRECTORY)
        st.video(processed_video_path)
    else:
        st.error("Недостаточно места для сохранения файла.")

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

