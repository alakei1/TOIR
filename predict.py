import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from utils.preprocessing import prepare_data
from utils.models import predict_anomalies
from utils.visualization import plot_results


def load_models():
    """Загрузка ранее обученных моделей"""
    print("Загрузка моделей...")
    lstm_model = load_model('models/lstm_model.h5')
    iforest = joblib.load('models/iforest_model.joblib')
    return {'lstm': lstm_model, 'iforest': iforest}


def run_prediction(input_csv, date_col='Дата'):
    # 1. Загрузка новых данных
    print(f"Загрузка данных из {input_csv}...")
    df = pd.read_csv(input_csv)
    processed_data, timestamps = prepare_data(df)

    # 2. Загрузка моделей
    models = load_models()

    # 3. Предсказание аномалий
    print("Предсказание...")
    predictions = predict_anomalies(models, processed_data)

    # 4. Визуализация
    print("Визуализация результатов...")
    plot_results(df, predictions, date_col=date_col)

    # 5. Вывод результатов
    print("\nОбнаруженные аномалии:")
    for model_name, preds in predictions.items():
        print(f"{model_name}: {sum(preds)} аномалий")

    print("\nГотово! Графики сохранены в папке 'results'")


if __name__ == "__main__":
    import argparse

    run_prediction("data/raw/sensor_data.csv", date_col="Дата")
    parser = argparse.ArgumentParser(description="Предсказание аномалий на новых данных")
    parser.add_argument('--input', type=str, required=True, help='Путь к CSV-файлу')
    parser.add_argument('--date_col', type=str, default='Дата', help='Имя колонки с датой (по умолчанию: Дата)')
    args = parser.parse_args()

    run_prediction(args.input, args.date_col)
