import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


def prepare_data(df):
    """
    Улучшенная очистка и подготовка данных
    Возвращает:
    - processed_df: обработанные числовые данные
    - timestamps: временные метки (если есть)
    """
    # Сохраняем временные метки отдельно
    timestamp_cols = df.select_dtypes(include=['datetime', 'object']).columns
    timestamps = df[timestamp_cols].copy() if not timestamp_cols.empty else None

    # Работаем только с числовыми данными
    numeric_df = df.select_dtypes(include=['number'])

    # Удаление полностью пустых колонок
    numeric_df = numeric_df.dropna(axis=1, how='all')

    # Проверка, что остались данные
    if numeric_df.empty:
        raise ValueError("Нет числовых данных для обработки после удаления нечисловых колонок!")

    # Заполнение пропусков
    numeric_df = numeric_df.interpolate(method='linear', limit_direction='both')

    # Удаление строк с оставшимися пропусками
    numeric_df = numeric_df.dropna()

    # Нормализация данных
    scaler = MinMaxScaler()
    processed_data = pd.DataFrame(
        scaler.fit_transform(numeric_df),
        columns=numeric_df.columns,
        index=numeric_df.index
    )

    return processed_data, timestamps


def split_data(data, timestamps=None, test_size=0.2):
    """
    Упрощенное разделение данных
    """
    split_idx = int(len(data) * (1 - test_size))

    if timestamps is not None:
        return (
            data.iloc[:split_idx],
            data.iloc[split_idx:],
            timestamps.iloc[:split_idx],
            timestamps.iloc[split_idx:]
        )
    return data.iloc[:split_idx], data.iloc[split_idx:]

def prepare_labels(df, contamination=0.05, random_state=42):
    """
    Улучшенная автоматическая разметка аномалий
    с обработкой возможных ошибок
    """
    try:
        # Проверка входных данных
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Выбираем предположительно нормальные периоды
        normal_data = df.sample(frac=0.1, random_state=random_state)

        # Проверка, что достаточно данных для обучения
        if len(normal_data) < 2:
            raise ValueError("Недостаточно данных для обучения детектора аномалий")

        # Создание и обучение модели
        clf = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        clf.fit(normal_data)

        # Предсказание аномалий
        labels = clf.predict(df)
        return (labels == -1).astype(int)

    except Exception as e:
        print(f"Ошибка при автоматической разметке: {str(e)}")
        # Возвращаем нулевые метки в случае ошибки
        return np.zeros(len(df), dtype=int)