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


# Добавляем текущую директорию в путь поиска модулей
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_directory)

def process(image_or_video):
    model = YOLO('best.pt')
    results = model(image_or_video, stream=True)
    
    # Обработка видео
    if isinstance(image_or_video, str) and image_or_video.endswith(('.mp4', '.avi')):
        output_path = tempfile.mktemp(suffix='.mp4')
        results.save(output_path)
        return output_path, results

    # Обработка изображения
    else:
        result_image = results.render()[0]
        pil_image = Image.fromarray(result_image)
        return pil_image, results

def remove_old_files(directory, age_limit=1800):  # 1800 секунд = 30 минут
    current_time = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            creation_time = os.path.getctime(file_path)
            if (current_time - creation_time) > age_limit:
                os.remove(file_path)
                print(f"Removed old file: {filename}")

st.title('Evodrone test')

remove_old_files("path/to/processed_videos")

# Загрузка и обработка изображения
image_file = st.file_uploader('Upload an image', type=['png', 'jpg'])
if image_file is not None:
    with st.spinner('Processing...'):
        image, results = process(image_file)
        st.image(image, caption='Processed Image')
        result_df = results.pandas().xyxy[0]
        result_json = result_df.to_json(orient="records")
        result_txt = result_df.to_string()
        result_csv = result_df.to_csv()
        st.json(result_json)
        st.download_button('Download JSON', result_json, file_name='results.json')
        st.text(result_txt)
        st.download_button('Download Text', result_txt, file_name='results.txt')
        st.download_button('Download CSV', result_csv, file_name='results.csv')

# Загрузка и обработка видео
video_file = st.file_uploader('Upload a video', type=['mp4'])
if video_file is not None:
    with st.spinner('Processing...'):
        video_path, _ = process(video_file)
        st.video(video_path)
        st.download_button('Download Processed Video', video_path, file_name='processed_video.mp4')
        st.caption("Note: Files are stored for only 30 minutes.")

# Загрузка и обработка видео с YouTube
youtube_url = st.text_input('Enter a YouTube URL')
if youtube_url:
    with st.spinner('Downloading and Processing...'):
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(file_extension='mp4').first()
        video_path = stream.download()
        processed_video_path, _ = process(video_path)
        st.video(processed_video_path)
        st.download_button('Download Processed Video', processed_video_path, file_name='processed_video.mp4')
        st.caption("Note: Files are stored for only 30 minutes.")



