import numpy as np
import pandas as pd
from utils.preprocessing import prepare_data, split_data
from utils.models import train_models, predict_anomalies
from utils.visualization import plot_results
from sklearn.ensemble import IsolationForest


def main():
    # 1. Загрузка и подготовка данных
    print("Шаг 1: Подготовка данных...")
    df = pd.read_csv('data/raw/sensor_data.csv')
    processed_data, timestamps = prepare_data(df)  # Распаковываем кортеж

    # 2. Разделение данных
    print("\nШаг 2: Разделение данных...")
    if timestamps is not None:
        train_data, test_data, train_timestamps, test_timestamps = split_data(
            processed_data, timestamps
        )
    else:
        train_data, test_data = split_data(processed_data)

    print("Проверка данных перед обучением:")
    print(f"Тип данных: {type(train_data)}")
    print(f"Форма данных: {train_data.shape}")
    print(f"Тип элементов: {type(train_data.iloc[0, 0])}")

    # 3. Автоматическая разметка данных
    print("\nАвтоматическая разметка данных...")
    # Убедимся, что данные чисто числовые
    train_data_numeric = train_data.select_dtypes(include=[np.number])
    train_normal = train_data_numeric.sample(frac=0.1)

    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(train_normal)
    train_labels = clf.predict(train_data_numeric)
    train_labels = (train_labels == -1).astype(int)

    # 4. Обучение моделей (передаем метки)
    print("\nШаг 3: Обучение моделей...")
    models = train_models(train_data_numeric, train_labels)  # Передаем размеченные данные
    predictions = predict_anomalies(models, test_data, lstm_threshold=0.3)


    predictions = predict_anomalies(models, test_data, lstm_threshold=0.3)

    print("\nРезультаты обнаружения аномалий:")
    print(f"LSTM (порог 0.3): {sum(predictions['lstm'])} аномалий")
    print(f"Isolation Forest: {sum(predictions['iforest'])} аномалий")
    print(f"Ансамбль: {sum(predictions['ensemble'])} аномалий")

    plot_results(test_data, predictions, date_col='Дата')

    # 5. Предсказание аномалий
    print("\nШаг 4: Обнаружение аномалий...")
    test_data_numeric = test_data.select_dtypes(include=[np.number])
    predictions = predict_anomalies(models, test_data_numeric)

    print("\nПроверка форматов перед визуализацией:")
    print(f"Тип test_data: {type(test_data)}")
    print(f"Тип predictions: {type(predictions)}")
    print("Ключи predictions:", predictions.keys())
    print("Форма LSTM предсказаний:", predictions['lstm'].shape)
    print("Форма IF предсказаний:", predictions['iforest'].shape)

    print("\nРезультаты обнаружения аномалий:")
    for model_name, preds in predictions.items():
        print(f"{model_name}: обнаружено {sum(preds)} аномалий")

    # 6. Визуализация (передаем оригинальные данные для меток времени)
    print("\nШаг 5: Визуализация результатов...")
    plot_results(test_data, predictions)  # test_data содержит временные метки

    print("\nГотово! Результаты сохранены в папке 'results'")


if __name__ == "__main__":
    main()