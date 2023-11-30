import sys
sys.path.insert(0, 'https://github.com/NataGoto/my_first_app1/blob/main/builder.py')
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os

# Добавляем текущую директорию в путь поиска модулей
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_directory)

# Теперь можно импортировать другие модули, включая builder.py

# Интеграция функции process прямо в app.py
def process(image_file):
    MODEL_NAME = 'model_air.h5'  # Убедитесь, что путь к модели указан правильно
    model = load_model(MODEL_NAME)
    INPUT_SHAPE = (256, 456, 3)

    image = Image.open(image_file)
    resized_image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))
    array = np.array(resized_image)[..., :3][np.newaxis, ...]
    prediction_array = (255 * model.predict(array)).astype(int)
    prediction_array = np.split(prediction_array, 2, axis=-1)[0]
    zeros = np.zeros_like(prediction_array)
    ones = np.ones_like(prediction_array)
    prediction_array_4d = np.concatenate([255 * (prediction_array > 100), zeros, zeros, 128 * ones], axis=3)[0].astype(np.uint8)
    mask_image = Image.fromarray(prediction_array_4d).resize(image.size)
    image.paste(mask_image, (0, 0), mask_image)
    return resized_image, prediction_array, image

# Код приложения Streamlit
st.title('Images segmentation demo')
image_file = st.file_uploader('Load an image', type=['png', 'jpg'])

if image_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(image_file)
    results = process(image_file)
    col1.text('Source image')
    col1.image(results[0])
    col2.text('Mask')
    col2.image(results[1])
    st.image(results[2])
