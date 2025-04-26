"""
Усовершенствованный модуль интеграции ML для ScalpMaster (SM)
Объединяет стандартные подходы с LSTM и расширенными признаками
"""

import logging
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import traceback
import threading

# Импорт модулей ScalpMaster
from ml_integration import MLIntegration
from pattern_detection import detect_patterns
from advanced_features import AdvancedFeatureExtractor
from lstm_analyzer import LSTMAnalyzer

# Настройка логирования
logger = logging.getLogger("ml_enhanced_integration")
if not logger.handlers:
    file_handler = logging.FileHandler("sm_enhanced_ml.log", encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

class EnhancedMLIntegration(MLIntegration):
    def __init__(self, model_dir="ml_models", data_dir="ml_data"):
        """
        Инициализация усовершенствованного модуля интеграции ML
        
        Args:
            model_dir: директория моделей
            data_dir: директория данных
        """
        # Инициализируем базовый класс
        super().__init__(model_dir=model_dir, data_dir=data_dir)
        
        # Инициализируем новые компоненты
        self.feature_extractor = AdvancedFeatureExtractor()
        self.lstm_analyzer = LSTMAnalyzer(model_dir=os.path.join(model_dir, "lstm"))
        
        # Настройки для LSTM
        self.use_lstm = True
        self.lstm_weight = 0.3  # Вес LSTM в итоговом прогнозе
        
        # Статистика LSTM
        self.lstm_stats = {
            "total_predictions": 0,
            "lstm_improved": 0,
            "lstm_worsened": 0
        }
        
        logger.info("Усовершенствованный модуль интеграции ML инициализирован")
    
    def toggle_lstm(self, enabled=True):
        """
        Включение/отключение использования LSTM
        
        Args:
            enabled: использовать LSTM (True/False)
        """
        self.use_lstm = enabled
        logger.info(f"LSTM {'включен' if enabled else 'отключен'}")
    
    def set_lstm_weight(self, weight):
        """
        Установка веса LSTM в итоговом прогнозе
        
        Args:
            weight: вес LSTM (0.0-1.0)
        """
        if 0.0 <= weight <= 1.0:
            self.lstm_weight = weight
            logger.info(f"Установлен вес LSTM: {weight}")
        else:
            logger.warning(f"Некорректный вес LSTM: {weight}. Допустимый диапазон: 0.0-1.0")
    
    def analyze_pattern(self, candles, conventional_results, currency_pair=None):
        """
        Анализ паттерна с использованием комбинации традиционного алгоритма, ML и LSTM
        
        Args:
            candles: список обнаруженных свечей
            conventional_results: результаты обычных алгоритмов (direction, confidence, info)
            currency_pair: валютная пара (опционально)
        
        Returns:
            tuple: (direction, confidence, info, is_enhanced)
        """
        direction, confidence, info = conventional_results
        
        # Если не используем ML или паттерн не определен, вернуть обычные результаты
        if not self.use_ml or not direction:
            return direction, confidence, info, False
        
        try:
            # 1. Получаем расширенные признаки
            enhanced_features = self.feature_extractor.calculate_advanced_features(candles, pair=currency_pair)
            
            # 2. Получаем прогноз от базового ML (из родительского класса)
            ml_direction, ml_confidence, ml_info, is_ml_used = super().analyze_pattern(candles, conventional_results)
            
            # 3. Если используем LSTM, получаем его прогноз
            lstm_direction, lstm_confidence, lstm_info = None, 0.0, {}
            if self.use_lstm and len(candles) >= 15:  # Минимальная длина для LSTM
                pair_key = currency_pair.replace("/", "_") if currency_pair else "EUR_USD"
                lstm_direction, lstm_confidence, lstm_info = self.lstm_analyzer.predict(candles, pair=pair_key)
            
            # 4. Объединяем все прогнозы
            final_direction, final_confidence, combined_info = self._combine_predictions(
                conventional=(direction, confidence),
                ml=(ml_direction, ml_confidence) if is_ml_used else (None, 0.0),
                lstm=(lstm_direction, lstm_confidence) if lstm_direction else (None, 0.0),
                enhanced_features=enhanced_features
            )
            
            # 5. Добавляем всю информацию в результат
            combined_info.update({
                "conventional_confidence": confidence,
                "ml_confidence": ml_confidence if is_ml_used else 0.0,
                "lstm_confidence": lstm_confidence if lstm_direction else 0.0,
                "enhanced_features": enhanced_features
            })
            
            # Обновляем статистику LSTM
            if lstm_direction:
                self.lstm_stats["total_predictions"] += 1
            
            return final_direction, final_confidence, combined_info, True
            
        except Exception as e:
            logger.error(f"Ошибка при усовершенствованном ML-анализе: {e}")
            logger.error(traceback.format_exc())
            return direction, confidence, info, False
    
    def _combine_predictions(self, conventional, ml, lstm, enhanced_features):
        """
        Объединение прогнозов из разных источников
        
        Args:
            conventional: (direction, confidence) из традиционного алгоритма
            ml: (direction, confidence) из ML
            lstm: (direction, confidence) из LSTM
            enhanced_features: расширенные признаки
            
        Returns:
            tuple: (direction, confidence, combined_info)
        """
        conv_dir, conv_conf = conventional
        ml_dir, ml_conf = ml
        lstm_dir, lstm_conf = lstm
        
        # Преобразуем направления в числовые значения для взвешивания
        # 1 = вверх, -1 = вниз, 0 = нет сигнала
        conv_val = 1 if conv_dir == "up" else -1 if conv_dir == "down" else 0
        ml_val = 1 if ml_dir == "up" else -1 if ml_dir == "down" else 0
        lstm_val = 1 if lstm_dir == "up" else -1 if lstm_dir == "down" else 0
        
        # Базовые веса
        conv_weight = self.conventional_weight
        ml_weight = self.ml_weight
        lstm_weight = self.lstm_weight
        
        # Корректируем веса в зависимости от наличия прогнозов
        valid_components = 0
        if conv_val != 0:
            valid_components += 1
        if ml_val != 0:
            valid_components += 1
        if lstm_val != 0:
            valid_components += 1
        
        # Если присутствуют не все компоненты, перераспределяем веса
        if valid_components < 3 and valid_components > 0:
            total_weight = conv_weight + ml_weight + lstm_weight
            if conv_val == 0:
                ml_weight += conv_weight * (ml_val != 0) / max(1, (ml_val != 0) + (lstm_val != 0))
                lstm_weight += conv_weight * (lstm_val != 0) / max(1, (ml_val != 0) + (lstm_val != 0))
                conv_weight = 0
            if ml_val == 0:
                conv_weight += ml_weight * (conv_val != 0) / max(1, (conv_val != 0) + (lstm_val != 0))
                lstm_weight += ml_weight * (lstm_val != 0) / max(1, (conv_val != 0) + (lstm_val != 0))
                ml_weight = 0
            if lstm_val == 0:
                conv_weight += lstm_weight * (conv_val != 0) / max(1, (conv_val != 0) + (ml_val != 0))
                ml_weight += lstm_weight * (ml_val != 0) / max(1, (conv_val != 0) + (ml_val != 0))
                lstm_weight = 0
        
        # Используем повышенный вес для LSTM при низком уровне шума
        if lstm_val != 0 and "noise_level" in enhanced_features:
            noise_level = enhanced_features["noise_level"]
            # Если низкий уровень шума, увеличиваем вес LSTM
            if noise_level < 0.02:  # Порог низкого шума
                noise_factor = 1.0 - min(1.0, noise_level / 0.02)
                lstm_boost = noise_factor * 0.1  # Максимальное повышение веса LSTM на 0.1
                
                # Забираем вес у других компонентов пропорционально их текущим весам
                total_others = conv_weight + ml_weight
                if total_others > 0:
                    conv_reduction = lstm_boost * (conv_weight / total_others)
                    ml_reduction = lstm_boost * (ml_weight / total_others)
                    
                    conv_weight -= conv_reduction
                    ml_weight -= ml_reduction
                    lstm_weight += lstm_boost
        
        # Вычисляем взвешенную сумму направлений
        weighted_direction = (
            conv_val * conv_weight +
            ml_val * ml_weight +
            lstm_val * lstm_weight
        )
        
        # Определяем финальное направление
        if weighted_direction > 0:
            final_direction = "up"
        elif weighted_direction < 0:
            final_direction = "down"
        else:
            # В случае равенства используем направление с наибольшей уверенностью
            confidences = [
                (conv_dir, conv_conf),
                (ml_dir, ml_conf),
                (lstm_dir, lstm_conf)
            ]
            
            # Оставляем только ненулевые прогнозы
            valid_confidences = [(d, c) for d, c in confidences if d is not None]
            
            if valid_confidences:
                final_direction = max(valid_confidences, key=lambda x: x[1])[0]
            else:
                final_direction = None
        
        # Вычисляем финальную уверенность
        # Если все компоненты согласны, увеличиваем уверенность
        agreement_boost = 0.0
        directions = [d for d in [conv_dir, ml_dir, lstm_dir] if d is not None]
        if len(directions) > 1:
            agreement = all(d == directions[0] for d in directions)
            if agreement:
                agreement_boost = 0.1 * (len(directions) - 1)  # Повышаем уверенность на 10% за каждое согласие
        
        final_confidence = (
            conv_conf * conv_weight +
            ml_conf * ml_weight +
            lstm_conf * lstm_weight
        )
        
        # Добавляем бонус за согласие
        final_confidence = min(0.95, final_confidence + agreement_boost)
        
        # Дополнительная информация
        combined_info = {
            "conventional_weight": float(conv_weight),
            "ml_weight": float(ml_weight),
            "lstm_weight": float(lstm_weight),
            "weighted_direction": float(weighted_direction),
            "agreement_boost": float(agreement_boost),
            "noise_level": float(enhanced_features.get("noise_level", 0)),
            "market_session": self._get_session_name(enhanced_features)
        }
        
        return final_direction, final_confidence, combined_info
    
    def _get_session_name(self, features):
        """Извлечение названия активной торговой сессии из расширенных признаков"""
        sessions = ["asia", "europe", "us", "overlap", "closed"]
        for session in sessions:
            if features.get(f"session_{session}", 0) == 1:
                return session
        return "unknown"
    
    def update_ml_from_feedback(self, pattern, features, is_correct):
        """
        Обновление ML-моделей и статистики на основе обратной связи
        
        Args:
            pattern: название паттерна
            features: признаки
            is_correct: был ли прогноз верным
        """
        # Вызываем метод базового класса
        super().update_ml_from_feedback(pattern, features, is_correct)
        
        # Обновляем статистику LSTM, если он был использован
        if self.use_lstm and "lstm_confidence" in features and features["lstm_confidence"] > 0:
            lstm_dir = "up" if features.get("lstm_value", 0) > 0 else "down"
            conv_dir = "up" if features.get("conventional_value", 0) > 0 else "down"
            
            # Проверяем, улучшил или ухудшил LSTM прогноз
            if is_correct and lstm_dir == conv_dir:
                # Оба дали правильный прогноз
                pass
            elif is_correct and lstm_dir != conv_dir:
                # LSTM дал правильный прогноз, традиционный алгоритм - нет
                self.lstm_stats["lstm_improved"] += 1
            elif not is_correct and lstm_dir == conv_dir:
                # Оба дали неправильный прогноз
                pass
            elif not is_correct and lstm_dir != conv_dir:
                # LSTM дал неправильный прогноз, традиционный алгоритм - верный
                self.lstm_stats["lstm_worsened"] += 1
    
    def get_enhanced_stats(self):
        """
        Получение расширенной статистики использования ML и LSTM
        
        Returns:
            dict: расширенная статистика
        """
        # Получаем базовую статистику
        stats = self.get_ml_stats()
        
        # Добавляем статистику LSTM
        stats["lstm"] = self.lstm_stats.copy()
        
        if stats["lstm"]["total_predictions"] > 0:
            stats["lstm"]["improved_percent"] = (stats["lstm"]["lstm_improved"] / stats["lstm"]["total_predictions"]) * 100
            stats["lstm"]["worsened_percent"] = (stats["lstm"]["lstm_worsened"] / stats["lstm"]["total_predictions"]) * 100
        
        return stats
    
    def train_lstm_model(self, candles_history, pair="EUR_USD", epochs=100):
        """
        Обучение LSTM модели на исторических данных
        
        Args:
            candles_history: список исторических свечей
            pair: валютная пара
            epochs: количество эпох обучения
            
        Returns:
            dict: результаты обучения
        """
        try:
            if len(candles_history) < 100:
                logger.warning(f"Недостаточно данных для обучения LSTM: {len(candles_history)} < 100")
                return {"status": "error", "message": "Недостаточно данных для обучения"}
            
            # Нормализуем имя пары
            pair_key = pair.replace("/", "_") if pair else "EUR_USD"
            
            # Обучаем LSTM модель
            result = self.lstm_analyzer.train_model(candles_history, pair=pair_key, epochs=epochs)
            
            if result["status"] == "success":
                logger.info(f"LSTM модель для пары {pair} успешно обучена. Точность: {result['accuracy']:.2f}")
            else:
                logger.warning(f"Ошибка при обучении LSTM модели: {result.get('message', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            error_msg = f"Ошибка при обучении LSTM модели: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}
    
    def schedule_lstm_retraining(self, interval_hours=24):
        """
        Планирование периодического переобучения LSTM моделей
        
        Args:
            interval_hours: интервал переобучения в часах
        """
        def retraining_job():
            while True:
                # Ожидание указанного времени
                time.sleep(interval_hours * 3600)
                
                logger.info("Запуск планового переобучения LSTM моделей...")
                try:
                    # Здесь можно использовать исторические данные из логов или внешний источник
                    # Сейчас делаем заглушку - в реальной системе нужно загрузить исторические данные
                    logger.info("Функция переобучения LSTM требует исторических данных")
                    
                except Exception as e:
                    logger.error(f"Ошибка при плановом переобучении LSTM: {e}")
                    logger.error(traceback.format_exc())
        
        # Запуск фоновой задачи переобучения
        retraining_thread = threading.Thread(target=retraining_job, daemon=True)
        retraining_thread.start()
        
        logger.info(f"Запланировано периодическое переобучение LSTM моделей каждые {interval_hours} часов")
