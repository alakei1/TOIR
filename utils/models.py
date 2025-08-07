import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def create_lstm_model(input_shape):
    """Создание LSTM модели для обнаружения аномалий"""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_models(train_data, train_labels=None):
    """
    Обучение всех моделей и сохранение на диск
    :param train_data: Обучающие данные (DataFrame или numpy array)
    :param train_labels: Метки (если None, будут созданы автоматически)
    :return: Словарь обученных моделей
    """
    # Преобразование в numpy array
    train_array = np.asarray(train_data).astype('float32')

    # Если метки не предоставлены, создаем фиктивные
    if train_labels is None:
        train_labels = np.zeros(len(train_array))
        print("Предупреждение: используются фиктивные метки")

    # 1. Обучение LSTM
    X_train_lstm = train_array.reshape(-1, 1, train_array.shape[1])
    lstm_model = create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))

    # Балансировка классов (если аномалий мало)
    class_weight = {0: 1, 1: 20} if sum(train_labels) < 0.1 * len(train_labels) else None

    lstm_model.fit(
        X_train_lstm,
        train_labels,
        epochs=15,
        batch_size=32,
        class_weight=class_weight,
        verbose=1
    )

    # 2. Обучение Isolation Forest
    iforest = IsolationForest(contamination=0.05, random_state=42)
    iforest.fit(train_array)

    # 3. Сохраняем модели
    os.makedirs('models', exist_ok=True)
    lstm_model.save('models/lstm_model.h5')
    joblib.dump(iforest, 'models/iforest_model.joblib')
    print("Модели сохранены в папке 'models/'")

    return {
        'lstm': lstm_model,
        'iforest': iforest
    }


def predict_anomalies(models, test_data, lstm_threshold=0.3):
    test_array = np.asarray(test_data).astype('float32')
    predictions = {}

    # LSTM
    lstm_input = test_array.reshape(-1, 1, test_array.shape[1])
    lstm_probs = models['lstm'].predict(lstm_input, verbose=0).flatten()
    predictions['lstm'] = (lstm_probs > lstm_threshold).astype(int)

    # Isolation Forest
    predictions['iforest'] = (models['iforest'].predict(test_array) == -1).astype(int)

    # Совместное решение
    predictions['ensemble'] = (
            (predictions['lstm'] + predictions['iforest']) >= 1
    ).astype(int)

    return predictions