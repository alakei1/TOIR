

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.dates import DateFormatter, HourLocator
from matplotlib.dates import AutoDateLocator, AutoDateFormatter



def plot_results(test_data, predictions, date_col=None):
    """
    Универсальная визуализация результатов обнаружения аномалий
    :param test_data: DataFrame с тестовыми данными
    :param predictions: Словарь с предсказаниями {'lstm': ..., 'iforest': ...}
    :param date_col: Название столбца с датами (None для автоматического определения)
    """
    try:
        # Проверка входных данных
        if not isinstance(test_data, pd.DataFrame):
            raise ValueError("test_data должен быть pandas DataFrame")

        # Создаем папку для результатов
        os.makedirs('results', exist_ok=True)

        # Определение столбца с датами
        datetime_col = None
        possible_date_cols = ['Дата', 'date', 'timestamp', 'time', 'время']

        if date_col is not None:
            if date_col in test_data.columns:
                datetime_col = date_col
            else:
                raise ValueError(
                    f"Указанный столбец с датами '{date_col}' не найден. Доступные столбцы: {list(test_data.columns)}")
        else:
            # Проверяем индекс
            if isinstance(test_data.index, pd.DatetimeIndex):
                test_data = test_data.reset_index()  # Преобразуем индекс в столбец
                datetime_col = 'index'
            else:
                # Ищем подходящий столбец
                for col in possible_date_cols:
                    if col in test_data.columns:
                        datetime_col = col
                        break

        if datetime_col is None:
            # Проверяем все столбцы на тип datetime
            for col in test_data.columns:
                if pd.api.types.is_datetime64_any_dtype(test_data[col]):
                    datetime_col = col
                    break

        if datetime_col is None:
            raise ValueError(f"Не найден столбец с датами. Проверенные имена: {possible_date_cols}")

        # Преобразуем даты
        dates = pd.to_datetime(test_data[datetime_col])

        # Получаем числовые столбцы (исключая даты)
        sensor_cols = [col for col in test_data.columns
                       if col != datetime_col and pd.api.types.is_numeric_dtype(test_data[col])]

        if not sensor_cols:
            raise ValueError("Не найдены числовые столбцы с данными сенсоров")

        # Создаем графики
        fig, axes = plt.subplots(
            nrows=len(sensor_cols),
            ncols=1,
            figsize=(15, 3 * len(sensor_cols)),
            squeeze=False
        )
        axes = axes.flatten()


        # Настройка стиля
        plt.style.use('seaborn-v0_8-darkgrid')  # <- заменил стиль
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Рисуем графики для каждого сенсора
        for i, sensor_col in enumerate(sensor_cols):
            ax = axes[i]

            # Основной график данных
            ax.plot(dates, test_data[sensor_col],
                    label=f'{sensor_col}',
                    color=colors[0],
                    linewidth=1,
                    alpha=0.7)

            # Аномалии LSTM
            if 'lstm' in predictions:
                lstm_anoms = np.where(predictions['lstm'] == 1)[0]
                ax.scatter(
                    dates[lstm_anoms],
                    test_data[sensor_col].iloc[lstm_anoms],
                    color=colors[1], marker='o', s=50,
                    alpha=0.7, label='LSTM аномалии'
                )

            # Аномалии Isolation Forest
            if 'iforest' in predictions:
                if_anoms = np.where(predictions['iforest'] == 1)[0]
                ax.scatter(
                    dates[if_anoms],
                    test_data[sensor_col].iloc[if_anoms],
                    color=colors[2], marker='x', s=50,
                    alpha=0.7, label='IF аномалии'
                )

            # Общие аномалии
            if 'lstm' in predictions and 'iforest' in predictions:
                common_anoms = np.where((predictions['lstm'] == 1) &
                                        (predictions['iforest'] == 1))[0]
                ax.scatter(
                    dates[common_anoms],
                    test_data[sensor_col].iloc[common_anoms],
                    color=colors[3], marker='*', s=100,
                    alpha=1.0, label='Общие аномалии'
                )

            # Настройка внешнего вида
            ax.set_title(f'Аномалии для {sensor_col}')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.6)

            # Форматирование дат

            locator = AutoDateLocator(minticks=3, maxticks=8)
            formatter = AutoDateFormatter(locator)

            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Сохраняем в разные форматы
        plt.savefig('results/anomalies.png', dpi=300, bbox_inches='tight')
        plt.savefig('results/anomalies.pdf', bbox_inches='tight')
        plt.close()

        print("Графики успешно сохранены в папке results/")

    except Exception as e:
        print(f"Ошибка при визуализации: {str(e)}")
        if 'test_data' in locals():
            print("Доступные столбцы:", test_data.columns.tolist())
            print("Пример данных:")
            print(test_data.head())
        raise