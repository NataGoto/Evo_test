
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

# Define path to the image file
source = uploaded_image

# Run inference on the source
results = model(source)  # list of Results objects

# Загрузка видео
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"], key='video_uploader')

# Define path to the image file
source = uploaded_video

# Run inference on the source
results = model(source)  # list of Results objects

# Ввод YouTube URL
youtube_url = st.text_input('Enter a YouTube URL', key='youtube_url_input')

# Define path to the image file
source = youtube_url

# Run inference on the source
results = model(source)  # list of Results objects

