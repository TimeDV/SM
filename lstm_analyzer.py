"""
LSTM модуль для анализа временных рядов в ScalpMaster (SM)
Обеспечивает работу с последовательностями данных и прогнозирование трендов
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# Настройка логирования
logger = logging.getLogger("lstm_analyzer")
if not logger.handlers:
    file_handler = logging.FileHandler("sm_lstm_analyzer.log", encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

class LSTMAnalyzer:
    def __init__(self, model_dir="lstm_models"):
        """
        Инициализация анализатора временных рядов на основе LSTM
        
        Args:
            model_dir: директория для хранения моделей
        """
        self.model_dir = model_dir
        self.models = {}  # словарь моделей для разных валютных пар
        self.scalers = {}  # словарь нормализаторов данных для разных валютных пар
        self.sequence_length = 15  # Длина последовательности (окно в 15 свечей, 45 минут при 3-минутном таймфрейме)
        self.prediction_horizon = 1  # Горизонт прогнозирования (на сколько свечей вперед)
        
        # Создаем директорию для моделей, если ее нет
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Загружаем существующие модели, если они есть
        self._load_models()
        
        logger.info("LSTM Analyzer инициализирован")
    
    def _load_models(self):
        """Загрузка сохраненных моделей из директории model_dir"""
        try:
            for filename in os.listdir(self.model_dir):
                if filename.endswith("_lstm_model.h5"):
                    pair = filename.split("_lstm_model.h5")[0]
                    model_path = os.path.join(self.model_dir, filename)
                    scaler_path = os.path.join(self.model_dir, f"{pair}_scaler.pkl")
                    
                    if os.path.exists(model_path) and os.path.exists(scaler_path):
                        self.models[pair] = load_model(model_path)
                        
                        with open(scaler_path, 'rb') as f:
                            self.scalers[pair] = pickle.load(f)
                        
                        logger.info(f"Модель LSTM для валютной пары '{pair}' успешно загружена")
            
            if not self.models:
                logger.info("Не найдено сохраненных LSTM моделей")
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке LSTM моделей: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _save_model(self, pair, model, scaler):
        """
        Сохранение обученной модели и нормализатора
        
        Args:
            pair: валютная пара
            model: обученная модель
            scaler: нормализатор данных
        """
        try:
            model_path = os.path.join(self.model_dir, f"{pair}_lstm_model.h5")
            scaler_path = os.path.join(self.model_dir, f"{pair}_scaler.pkl")
            
            # Сохраняем модель с использованием TensorFlow SavedModel format
            model.save(model_path)
            
            # Сохраняем скалер
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            logger.info(f"Модель LSTM для валютной пары '{pair}' успешно сохранена")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при сохранении LSTM модели: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _preprocess_candles(self, candles, normalize=True):
        """
        Предобработка свечей для использования в LSTM
        
        Args:
            candles: список обнаруженных свечей
            normalize: нормализовать ли данные
            
        Returns:
            np.array: предобработанные данные для LSTM
            dict: метаданные (среднее, стандартное отклонение и т.д.)
        """
        try:
            if len(candles) < self.sequence_length:
                logger.warning(f"Недостаточно свечей для LSTM анализа: {len(candles)} < {self.sequence_length}")
                return None, {}
            
            # Извлекаем признаки из свечей
            features = []
            
            for candle in candles:
                # Основные признаки свечи
                candle_features = [
                    candle["top"],  # Верхняя граница (High)
                    candle["bottom"],  # Нижняя граница (Low)
                    candle["center_y"],  # Центр свечи (примерно Close)
                    candle["height"],  # Высота свечи (волатильность)
                    1 if candle["color"] == "green" else -1 if candle["color"] == "red" else 0,  # Направление свечи
                ]
                
                # Дополнительные признаки можно добавить здесь
                
                features.append(candle_features)
            
            # Преобразуем в numpy массив
            features = np.array(features)
            
            # Создаем словарь с метаданными (полезными для интерпретации)
            metadata = {
                "mean": np.mean(features, axis=0),
                "std": np.std(features, axis=0),
                "min": np.min(features, axis=0),
                "max": np.max(features, axis=0),
                "last_value": features[-1].copy()
            }
            
            # Нормализация данных, если требуется
            if normalize:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                features = scaler.fit_transform(features)
                metadata["scaler"] = scaler
            
            # Создаем последовательности для LSTM
            sequences = []
            for i in range(len(features) - self.sequence_length + 1):
                sequences.append(features[i:i + self.sequence_length])
            
            # Возвращаем последовательности и метаданные
            return np.array(sequences), metadata
            
        except Exception as e:
            logger.error(f"Ошибка при предобработке свечей для LSTM: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, {}
    
    def _create_model(self, input_shape):
        """
        Создание модели LSTM с механизмом внимания
        
        Args:
            input_shape: форма входных данных (sequence_length, features)
            
        Returns:
            модель Keras
        """
        model = Sequential()
        
        # Первый слой LSTM (возвращаем последовательности для следующего слоя)
        model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh'), 
                              input_shape=input_shape))
        model.add(LayerNormalization())
        model.add(Dropout(0.2))
        
        # Второй слой LSTM
        model.add(Bidirectional(LSTM(32, return_sequences=False, activation='tanh')))
        model.add(LayerNormalization())
        model.add(Dropout(0.2))
        
        # Выходной слой (прогноз направления: вверх или вниз)
        model.add(Dense(1, activation='tanh'))  # tanh для значений от -1 до 1
        
        # Компиляция модели
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        
        return model
    
    def train_model(self, candles_history, pair="EUR_USD", epochs=100, batch_size=32, validation_split=0.2):
        """
        Обучение LSTM модели на исторических данных свечей
        
        Args:
            candles_history: список исторических свечей (должно быть много)
            pair: валютная пара
            epochs: количество эпох обучения
            batch_size: размер батча
            validation_split: доля данных для валидации
            
        Returns:
            dict: результаты обучения
        """
        try:
            # Предобработка данных
            sequences, metadata = self._preprocess_candles(candles_history)
            
            if sequences is None:
                return {"status": "error", "message": "Ошибка предобработки данных"}
            
            if len(sequences) < 50:  # Минимальное количество последовательностей для обучения
                logger.warning(f"Недостаточно данных для обучения: {len(sequences)} последовательностей")
                return {"status": "error", "message": f"Недостаточно данных для обучения: {len(sequences)} последовательностей"}
            
            # Подготовка данных для обучения
            # X - последовательности кроме последнего значения каждой последовательности
            # y - направление изменения цены (1: вверх, -1: вниз)
            X = sequences
            
            # Получаем центральные цены (центр свечи, индекс 2)
            # Для каждой последовательности, получаем последнюю цену и сравниваем с ценой через prediction_horizon
            y = []
            for i in range(len(candles_history) - self.sequence_length - self.prediction_horizon + 1):
                current_price = candles_history[i + self.sequence_length - 1]["center_y"]
                future_price = candles_history[i + self.sequence_length + self.prediction_horizon - 1]["center_y"]
                # Учитываем, что на экране Y растет вниз
                y.append(1.0 if current_price > future_price else -1.0)
            
            y = np.array(y)
            
            # Если нет достаточного количества меток
            if len(y) != len(X):
                logger.error(f"Несоответствие размеров X и y: {len(X)} vs {len(y)}")
                return {"status": "error", "message": f"Несоответствие размеров X и y: {len(X)} vs {len(y)}"}
            
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, shuffle=False)
            
            # Создание модели
            model = self._create_model((self.sequence_length, X.shape[2]))
            
            # Определяем коллбэки для обучения
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(filepath=os.path.join(self.model_dir, f"{pair}_checkpoint.h5"), 
                               save_best_only=True, monitor='val_loss')
            ]
            
            # Обучение модели
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Оценка на тестовой выборке
            evaluation = model.evaluate(X_test, y_test)
            
            # Сохраняем модель и скалер
            self.models[pair] = model
            self.scalers[pair] = metadata.get("scaler")
            self._save_model(pair, model, self.scalers[pair])
            
            # Формируем результаты
            results = {
                "status": "success",
                "accuracy": self._calculate_accuracy(model, X_test, y_test),
                "loss": evaluation[0],
                "mae": evaluation[1],
                "sample_size": len(sequences),
                "history": {
                    "loss": [float(x) for x in history.history['loss']],
                    "val_loss": [float(x) for x in history.history['val_loss']],
                    "mae": [float(x) for x in history.history['mae']],
                    "val_mae": [float(x) for x in history.history['val_mae']]
                }
            }
            
            logger.info(f"Модель LSTM для валютной пары '{pair}' обучена, точность: {results['accuracy']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при обучении LSTM модели: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}
    
    def _calculate_accuracy(self, model, X_test, y_test):
        """Вычисление точности прогнозов (доля правильно угаданных направлений)"""
        try:
            predictions = model.predict(X_test)
            # Преобразуем непрерывные значения в дискретные направления
            predicted_directions = (predictions > 0).astype(int) * 2 - 1  # -1 или 1
            true_directions = (y_test > 0).astype(int) * 2 - 1  # -1 или 1
            
            # Вычисляем процент совпадений
            accuracy = np.mean(predicted_directions.flatten() == true_directions)
            return float(accuracy)
        except Exception as e:
            logger.error(f"Ошибка при расчете точности: {e}")
            return 0.0
    
    def predict(self, candles, pair="EUR_USD"):
        """
        Прогнозирование направления движения цены с использованием LSTM
        
        Args:
            candles: список обнаруженных свечей
            pair: валютная пара
            
        Returns:
            tuple: (направление, уверенность, дополнительная информация)
        """
        try:
            # Проверяем наличие модели для данной валютной пары
            if pair not in self.models or pair not in self.scalers:
                logger.warning(f"Модель LSTM для валютной пары '{pair}' не найдена")
                return None, 0.0, {}
            
            # Получаем последовательность для прогнозирования
            if len(candles) < self.sequence_length:
                logger.warning(f"Недостаточно свечей для LSTM прогноза: {len(candles)} < {self.sequence_length}")
                return None, 0.0, {}
            
            # Извлекаем признаки из последних N свечей
            recent_candles = candles[-self.sequence_length:]
            features = []
            
            for candle in recent_candles:
                candle_features = [
                    candle["top"],
                    candle["bottom"],
                    candle["center_y"],
                    candle["height"],
                    1 if candle["color"] == "green" else -1 if candle["color"] == "red" else 0,
                ]
                features.append(candle_features)
            
            # Преобразуем в numpy массив
            features = np.array(features)
            
            # Нормализуем с использованием сохраненного скалера
            scaler = self.scalers[pair]
            normalized_features = scaler.transform(features)
            
            # Формируем входной тензор для модели
            X = np.array([normalized_features])
            
            # Получаем прогноз от модели
            prediction = self.models[pair].predict(X, verbose=0)[0][0]
            
            # Преобразуем результат в направление и уверенность
            # Значения ближе к 1 или -1 указывают на большую уверенность
            direction = "up" if prediction > 0 else "down"
            confidence = min(abs(prediction) * 1.0, 0.95)  # Ограничиваем уверенность 95%
            
            # Дополнительная информация для отладки и объяснения прогноза
            additional_info = {
                "raw_prediction": float(prediction),
                "normalized_confidence": float(confidence),
                "feature_count": features.shape[1],
                "sequence_length": self.sequence_length,
                "last_price": float(candles[-1]["center_y"]),
                "price_range": float(max(c["top"] for c in recent_candles) - min(c["bottom"] for c in recent_candles)),
                "recent_direction": "up" if candles[-1]["color"] == "green" else "down" if candles[-1]["color"] == "red" else "neutral"
            }
            
            logger.info(f"LSTM прогноз для {pair}: {direction} с уверенностью {confidence:.2f}")
            
            return direction, confidence, additional_info
            
        except Exception as e:
            logger.error(f"Ошибка при LSTM прогнозе: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, 0.0, {}
