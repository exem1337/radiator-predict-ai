from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import pickle
from typing import List
import os

app = FastAPI()

class RadiatorData(BaseModel):
    date: str
    aluminum_demand_increase: float
    copper_demand_increase: float

# Переменная для хранения пути к файлу модели
MODEL_FILE = 'trained_model.h5'

# Функция для создания мокового обучающего датасета
def create_mock_dataset():
    np.random.seed(42)
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    data = {
        'Дата': date_range,
        'Прирост спроса на алюминевый радиатор': np.random.randint(50, 200, size=len(date_range)),
        'Прирост спроса на медный радиатор': np.random.randint(40, 180, size=len(date_range)),
        'Тип радиатора': np.random.choice(['Алюминиевый', 'Медный'], size=len(date_range))
    }
    mock_dataset = pd.DataFrame(data)
    mock_dataset = mock_dataset.sort_values(by='Дата').reset_index(drop=True)
    return mock_dataset

# Функция для обучения и сохранения модели
def train_and_save_model():
    try:
        mock_dataset = create_mock_dataset()

        X = mock_dataset[['Прирост спроса на алюминевый радиатор', 'Прирост спроса на медный радиатор']].values
        y = mock_dataset['Тип радиатора'].apply(lambda x: 1 if x == 'Медный' else 0).values
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, input_shape=(2, 1)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Обучение модели
        history = model.fit(X, y, epochs=10, batch_size=32, verbose=1)

        # Сохранение модели в файл
        model.save('trained_model.h5')

        # Сохранение информации об обучении в файл
        with open('training_history.pkl', 'wb') as file:
            pickle.dump(history.history, file)

        print("Модель обучена и сохранена в файле 'trained_model.h5'")
        print("История обучения модели сохранена в файле 'training_history.pkl'")
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")

# Функция для загрузки модели
def load_saved_model():
    return load_model(MODEL_FILE)

class RadiatorDataList(BaseModel):
    data: List[RadiatorData]

def is_model_exist():
    return os.path.exists(MODEL_FILE)

def load_or_train_model():
    if not is_model_exist():
        train_and_save_model()

@app.post("/predict")
def predict_radiator(data: RadiatorDataList):
    try:
        # Преобразование JSON в DataFrame
        df = pd.DataFrame([item.dict() for item in data.data])

        model = load_saved_model()

        # Используем только нужные столбцы для предсказания
        prediction = model.predict(df[['aluminum_demand_increase', 'copper_demand_increase']])

        if prediction.mean() >= 0.5:
            return {"predicted_radiator_type": "Медный"}
        else:
            return {"predicted_radiator_type": "Алюминиевый"}
    except Exception as e:
        return {"error": f"Ошибка при предсказании: {e}"}

@app.get("/load_history")
def load_training_history():
    try:
        # Открытие файла с историей обучения
        with open('training_history.pkl', 'rb') as file:
            loaded_history = pickle.load(file)
        return {"loaded_history": loaded_history}
    except FileNotFoundError:
        return {"message": "Файл с историей обучения не найден"}
    except Exception as e:
        return {"message": f"Ошибка при загрузке истории обучения: {e}"}

@app.get("/model_info")
def get_model_info():
    try:
        load_or_train_model()
        model = load_saved_model()

        model_info = []

        # Проходим по слоям модели и собираем информацию о каждом слое
        for i, layer in enumerate(model.layers):
            layer_info = {
                "Индекс_слоя": i,
                "Название_слоя": layer.name,
                "Тип_слоя": layer.__class__.__name__,
                "Конфигурация_слоя": layer.get_config(),
                "Количество_параметров_слоя": layer.count_params()
            }
            model_info.append(layer_info)

        # Другая общая информация о модели
        model_summary = {
            "Всего_слоев": len(model.layers),
            "Всего_параметров": model.count_params(),
            "Конфигурация_модели": model.get_config()
        }

        return {
            "Сводка_модели": model_summary,
            "Информация_о_слоях": model_info
        }
    except Exception as e:
        return {"error": f"Ошибка при получении информации о модели: {e}"}
