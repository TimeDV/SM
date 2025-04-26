"""
ML Integration модуль для системы ScalpMaster (SM)
Обеспечивает интеграцию ML-моделей с основной системой анализа паттернов
"""

import logging
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import traceback
import threading

# Импорты модулей ScalpMaster
from pattern_detection import detect_patterns
from ml_pattern_analyzer import MLPatternAnalyzer
from ml_data_processor import MLDataProcessor

# Настройка логирования
logger = logging.getLogger("ml_integration")
if not logger.handlers:
    file_handler = logging.FileHandler("sm_ml_integration.log", encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

class MLIntegration:
    def __init__(self, model_dir="ml_models", data_dir="ml_data"):
        """
        Инициализация модуля интеграции ML
        
        Args:
            model_dir: директория моделей
            data_dir: директория данных
        """
        self.analyzer = MLPatternAnalyzer(model_dir=model_dir)
        self.data_processor = MLDataProcessor(data_dir=data_dir)
        
        # Настройки интеграции
        self.use_ml = True  # Использовать ML-модели
        self.ml_weight = 0.3  # Вес ML-модели в итоговом прогнозе (0.0-1.0)
        self.conventional_weight = 0.7  # Вес обычного алгоритма (0.0-1.0)
        self.min_ml_confidence = 0.6  # Минимальная уверенность ML для учета прогноза
        
        # Статистика
        self.stats = {
            "total_predictions": 0,
            "ml_improved": 0,  # Случаи когда ML улучшил прогноз
            "ml_worsened": 0,  # Случаи когда ML ухудшил прогноз
            "pattern_stats": {}  # Статистика по паттернам
        }
        
        # Обучение моделей если есть данные
        self._check_and_train_models()
        
        logger.info("Модуль ML Integration инициализирован")
    
    def _check_and_train_models(self):
        """Проверяет наличие моделей и при необходимости обучает их"""
        logger.info("Проверка наличия обученных моделей...")
        
        # Проверяем существование основных моделей
        missing_models = []
        for pattern in self.analyzer.pattern_names:
            model_path = os.path.join(self.analyzer.model_dir, f"{pattern.lower().replace(' ', '_')}_model.pkl")
            if not os.path.exists(model_path):
                missing_models.append(pattern)
        
        # Если есть отсутствующие модели, пробуем обучить
        if missing_models:
            logger.info(f"Отсутствуют модели для паттернов: {', '.join(missing_models)}")
            logger.info("Попытка обучения моделей...")
            
            # Обучаем модели если есть лог прогнозов
            if os.path.exists("pattern_forecast_log.csv"):
                result = self.analyzer.train_models(log_file="pattern_forecast_log.csv")
                if result["status"] == "success":
                    logger.info("Модели успешно обучены")
                else:
                    logger.warning(f"Проблема при обучении моделей: {result.get('message', 'Unknown error')}")
            else:
                logger.warning("Отсутствует файл логов для обучения моделей")
    
    def set_ml_weight(self, weight):
        """
        Установка веса ML-модели в итоговом прогнозе
        
        Args:
            weight: вес ML-модели (0.0-1.0)
        """
        if 0.0 <= weight <= 1.0:
            self.ml_weight = weight
            self.conventional_weight = 1.0 - weight
            logger.info(f"Установлен вес ML: {weight}, вес обычных алгоритмов: {1.0 - weight}")
        else:
            logger.warning(f"Некорректный вес ML: {weight}. Допустимый диапазон: 0.0-1.0")
    
    def toggle_ml(self, enabled=True):
        """
        Включение/отключение использования ML-моделей
        
        Args:
            enabled: использовать ML-модели (True/False)
        """
        self.use_ml = enabled
        logger.info(f"ML-модели {'включены' if enabled else 'отключены'}")
    
    def analyze_pattern(self, candles, conventional_results):
        """
        Анализ паттерна с использованием комбинации ML и обычных алгоритмов
        
        Args:
            candles: список обнаруженных свечей
            conventional_results: результаты обычных алгоритмов (pattern, confidence, info)
        
        Returns:
            tuple: (pattern, confidence, info, is_ml_used)
        """
        pattern, confidence, info = conventional_results
        
        # Если не используем ML или паттерн не определен, вернуть обычные результаты
        if not self.use_ml or not pattern:
            return pattern, confidence, info, False
        
        try:
            # Получаем прогноз от ML-модели
            features = self.data_processor.generate_additional_features(candles)
            ml_pattern, ml_confidence, ml_info, ml_proba = self.analyzer.analyze_candles(candles, conventional_results)
            
            # Если ML-уверенность ниже порога, используем только обычный алгоритм
            if ml_confidence < self.min_ml_confidence:
                logger.info(f"ML-уверенность ({ml_confidence:.2f}) ниже порога ({self.min_ml_confidence}), используем только обычный алгоритм")
                return pattern, confidence, info, False
            
            # Комбинируем результаты
            combined_confidence = (confidence * self.conventional_weight) + (ml_confidence * self.ml_weight)
            
            # Добавляем информацию о ML в результат
            combined_info = info.copy() if isinstance(info, dict) else {}
            combined_info["ml_confidence"] = ml_confidence
            combined_info["conventional_confidence"] = confidence
            combined_info["ml_probability"] = ml_proba
            combined_info["ml_weight"] = self.ml_weight
            
            # Логирование
            logger.info(f"Анализ паттерна: {pattern}")
            logger.info(f"Обычный алгоритм: {confidence:.2f}, ML: {ml_confidence:.2f}, Объединенная: {combined_confidence:.2f}")
            
            # Обновляем статистику
            self.stats["total_predictions"] += 1
            
            # Обновляем статистику по паттернам
            if pattern not in self.stats["pattern_stats"]:
                self.stats["pattern_stats"][pattern] = {
                    "total": 0,
                    "ml_improved": 0,
                    "ml_worsened": 0,
                    "avg_conventional_confidence": 0,
                    "avg_ml_confidence": 0,
                    "avg_combined_confidence": 0
                }
            
            pattern_stats = self.stats["pattern_stats"][pattern]
            pattern_stats["total"] += 1
            pattern_stats["avg_conventional_confidence"] = ((pattern_stats["avg_conventional_confidence"] * 
                                                          (pattern_stats["total"] - 1)) + confidence) / pattern_stats["total"]
            pattern_stats["avg_ml_confidence"] = ((pattern_stats["avg_ml_confidence"] * 
                                                 (pattern_stats["total"] - 1)) + ml_confidence) / pattern_stats["total"]
            pattern_stats["avg_combined_confidence"] = ((pattern_stats["avg_combined_confidence"] * 
                                                      (pattern_stats["total"] - 1)) + combined_confidence) / pattern_stats["total"]
            
            return pattern, combined_confidence, combined_info, True
            
        except Exception as e:
            logger.error(f"Ошибка при ML-анализе паттерна: {e}")
            logger.error(traceback.format_exc())
            return pattern, confidence, info, False
    
    def update_ml_from_feedback(self, pattern, features, is_correct):
        """
        Обновление ML-моделей на основе обратной связи
        
        Args:
            pattern: название паттерна
            features: признаки
            is_correct: был ли прогноз верным
        """
        if not self.use_ml:
            return
        
        try:
            self.analyzer.get_feedback(pattern, features, is_correct)
            
            # Обновляем статистику на основе результата
            if pattern in self.stats["pattern_stats"]:
                pattern_stats = self.stats["pattern_stats"][pattern]
                
                # Рассчитываем, улучшил ли ML прогноз (сложная логика зависит от проблемы)
                conventional_would_be_right = (is_correct and pattern_stats["avg_conventional_confidence"] > 0.7) or \
                                             (not is_correct and pattern_stats["avg_conventional_confidence"] < 0.3)
                
                ml_improved = (is_correct and pattern_stats["avg_ml_confidence"] > pattern_stats["avg_conventional_confidence"]) or \
                             (not is_correct and pattern_stats["avg_ml_confidence"] < pattern_stats["avg_conventional_confidence"])
                
                if ml_improved:
                    self.stats["ml_improved"] += 1
                    pattern_stats["ml_improved"] += 1
                elif conventional_would_be_right:
                    self.stats["ml_worsened"] += 1
                    pattern_stats["ml_worsened"] += 1
            
        except Exception as e:
            logger.error(f"Ошибка при обновлении ML из обратной связи: {e}")
            logger.error(traceback.format_exc())
    
    def schedule_model_retraining(self, interval_hours=24):
        """
        Планирование периодического переобучения моделей
        
        Args:
            interval_hours: интервал переобучения в часах
        """
        def retraining_job():
            while True:
                # Ожидание указанного времени
                time.sleep(interval_hours * 3600)
                
                logger.info("Запуск планового переобучения ML-моделей...")
                try:
                    # Переобучение моделей
                    result = self.analyzer.train_models(log_file="pattern_forecast_log.csv")
                    
                    if result["status"] == "success":
                        logger.info("Плановое переобучение успешно завершено")
                        
                        # Логируем результаты по каждому паттерну
                        for pattern, pattern_result in result.get("results", {}).items():
                            if pattern_result["status"] == "success":
                                logger.info(f"Паттерн '{pattern}': точность {pattern_result.get('accuracy', 0):.2f}, "
                                           f"примеров: {pattern_result.get('sample_size', 0)}")
                            else:
                                logger.warning(f"Паттерн '{pattern}': {pattern_result.get('message', 'Unknown error')}")
                    else:
                        logger.warning(f"Проблема при плановом переобучении: {result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Ошибка при плановом переобучении: {e}")
                    logger.error(traceback.format_exc())
        
        # Запуск фоновой задачи переобучения
        retraining_thread = threading.Thread(target=retraining_job, daemon=True)
        retraining_thread.start()
        
        logger.info(f"Запланировано периодическое переобучение моделей каждые {interval_hours} часов")
    
    def get_ml_stats(self):
        """
        Получение статистики использования ML
        
        Returns:
            dict: статистика использования ML
        """
        # Обогащаем статистику дополнительной информацией
        stats = self.stats.copy()
        
        if stats["total_predictions"] > 0:
            stats["ml_improved_percent"] = (stats["ml_improved"] / stats["total_predictions"]) * 100
            stats["ml_worsened_percent"] = (stats["ml_worsened"] / stats["total_predictions"]) * 100
            stats["ml_neutral_percent"] = 100 - stats["ml_improved_percent"] - stats["ml_worsened_percent"]
        
        return stats