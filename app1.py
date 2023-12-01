
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

