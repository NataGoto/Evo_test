

from tensorflow.keras.models import load_model
MODEL_NAME =   'model_air.h5'
import numpy as np
from PIL import Image
model = load_model(MODEL_NAME)                                              # Загрузка весов модели
INPUT_SHAPE = (256, 456, 3)


def process(image_file):
    image = Image.open(image_file)  # Открытие обрабатываемого файла
    resized_image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))          # Изменение размера изображения в соответствии со входом сети
    array = np.array(resized_image)[..., :3][np.newaxis, ..., np.newaxis]   # Регулировка формы тензора для подачи в сеть
    prediction_array = (255 * model.predict(array)).astype(int)             # Запуск предсказания сети
    prediction_array = np.split(prediction_array, 2, axis = -1)[0]          # Нулевой канал предсказания (значения 0 - самолет, 1 - фон)
    zeros = np.zeros_like(prediction_array)                                 # Создание массива нулей
    ones = np.ones_like(prediction_array)                                   # Создание массива единиц
    prediction_array_4d = np.concatenate([255 * (prediction_array > 100), zeros, zeros, 128 * ones], axis=3)[0].astype(np.uint8)  # Формирование тензора для наложения найденной маски
    mask_image = Image.fromarray(prediction_array_4d).resize(image.size)    # Преобразование тензора в изображение и подгонка его размера к исходному
    image.paste(mask_image, (0, 0), mask_image)                             # Добавление маски на исходное изображение
    return resized_image, prediction_array, image                           # Возврат исходного уменьшенного изображения, найденной маски и исходного изображения с наложенной маской
