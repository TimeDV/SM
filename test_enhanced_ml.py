"""
Тестирование усовершенствованных ML-компонентов ScalpMaster
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging

# Устанавливаем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импорт модулей ScalpMaster
from ml_enhanced_integration import EnhancedMLIntegration
from advanced_features import AdvancedFeatureExtractor
from lstm_analyzer import LSTMAnalyzer
from ml_visualization import MLVisualizer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler("test_enhanced_ml.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_enhanced_ml")

def generate_synthetic_candles(num_candles=100, trend="up"):
    """
    Генерация синтетических свечей для тестирования
    
    Args:
        num_candles: количество свечей
        trend: тренд цены ("up", "down", "sideways")
        
    Returns:
        list: список синтетических свечей
    """
    # Базовые параметры
    starting_price = 100
    volatility = 2.0
    candle_width = 10
    
    # Параметры тренда
    if trend == "up":
        trend_factor = -0.2  # Цена растет (Y уменьшается на экране)
    elif trend == "down":
        trend_factor = 0.2   # Цена падает (Y увеличивается на экране)
    else:
        trend_factor = 0.0   # Боковой тренд
    
    # Генерация цен закрытия с трендом и случайностью
    np.random.seed(42)  # Для воспроизводимости
    
    # Генерация ценового ряда
    closes = []
    for i in range(num_candles):
        # Добавляем тренд и случайность
        random_factor = np.random.normal(0, volatility)
        price = starting_price + trend_factor * i + random_factor
        closes.append(price)
    
    # Создание свечей
    candles = []
    for i in range(num_candles):
        # Определяем цвет свечи (растущая или падающая)
        if i > 0:
            color = "green" if closes[i] < closes[i-1] else "red"
            # На экране Y растет вниз, поэтому логика инвертирована
        else:
            color = "green" if np.random.random() > 0.5 else "red"
        
        # Высота свечи (волатильность)
        height = np.random.uniform(5, 15)
        
        # Координаты центра свечи
        center_x = i * candle_width
        center_y = closes[i]
        
        # Создаем "контур" (просто прямоугольник)
        contour = np.array([
            [center_x - candle_width // 2, center_y - height // 2],
            [center_x + candle_width // 2, center_y - height // 2],
            [center_x + candle_width // 2, center_y + height // 2],
            [center_x - candle_width // 2, center_y + height // 2]
        ], dtype=np.int32)
        
        # Добавляем свечу в список
        candle = {
            "center_x": center_x,
            "center_y": center_y,
            "top": center_y - height // 2,
            "bottom": center_y + height // 2,
            "width": candle_width,
            "height": height,
            "color": color,
            "contour": contour
        }
        candles.append(candle)
    
    return candles

def test_lstm_analyzer():
    """Тестирование LSTM-анализатора на синтетических данных"""
    print("\n--- Тестирование LSTM-анализатора ---")
    
    # Создаем экземпляр LSTM-анализатора
    lstm_analyzer = LSTMAnalyzer()
    
    # Генерируем синтетические данные для обучения
    print("Генерация синтетических данных для обучения...")
    training_candles = generate_synthetic_candles(num_candles=500, trend="up")
    
    # Обучаем модель на синтетических данных
    print("Обучение LSTM-модели...")
    train_result = lstm_analyzer.train_model(
        training_candles, 
        pair="TEST_PAIR", 
        epochs=20,
        batch_size=16
    )
    
    # Выводим результаты обучения
    if train_result["status"] == "success":
        print(f"Модель успешно обучена. Точность: {train_result['accuracy']:.2f}")
    else:
        print(f"Ошибка при обучении модели: {train_result.get('message', 'Unknown error')}")
        return
    
    # Генерируем тестовые данные
    print("\nГенерация тестовых данных...")
    test_candles_up = generate_synthetic_candles(num_candles=50, trend="up")
    test_candles_down = generate_synthetic_candles(num_candles=50, trend="down")
    
    # Тестируем модель на данных с восходящим трендом
    print("Тестирование на данных с восходящим трендом:")
    direction_up, confidence_up, info_up = lstm_analyzer.predict(test_candles_up, pair="TEST_PAIR")
    print(f"Прогноз: {direction_up}, уверенность: {confidence_up:.2f}")
    
    # Тестируем модель на данных с нисходящим трендом
    print("Тестирование на данных с нисходящим трендом:")
    direction_down, confidence_down, info_down = lstm_analyzer.predict(test_candles_down, pair="TEST_PAIR")
    print(f"Прогноз: {direction_down}, уверенность: {confidence_down:.2f}")
    
    # Проверяем корректность прогнозов
    expected_up = "up"
    expected_down = "down"
    
    if direction_up == expected_up:
        print("✓ Правильный прогноз для восходящего тренда")
    else:
        print(f"✗ Неправильный прогноз для восходящего тренда. Ожидалось: {expected_up}, получено: {direction_up}")
    
    if direction_down == expected_down:
        print("✓ Правильный прогноз для нисходящего тренда")
    else:
        print(f"✗ Неправильный прогноз для нисходящего тренда. Ожидалось: {expected_down}, получено: {direction_down}")

def test_advanced_features():
    """Тестирование модуля расширенных признаков"""
    print("\n--- Тестирование модуля расширенных признаков ---")
    
    # Создаем экземпляр извлекателя признаков
    feature_extractor = AdvancedFeatureExtractor()
    
    # Генерируем синтетические данные
    print("Генерация синтетических данных...")
    candles_up = generate_synthetic_candles(num_candles=50, trend="up")
    candles_down = generate_synthetic_candles(num_candles=50, trend="down")
    
    # Извлекаем расширенные признаки
    print("Извлечение расширенных признаков для восходящего тренда...")
    features_up = feature_extractor.calculate_advanced_features(candles_up)
    
    print("Извлечение расширенных признаков для нисходящего тренда...")
    features_down = feature_extractor.calculate_advanced_features(candles_down)
    
    # Выводим полученные признаки
    print("\nРасширенные признаки для восходящего тренда:")
    for key, value in features_up.items():
        print(f"{key}: {value}")
    
    print("\nРасширенные признаки для нисходящего тренда:")
    for key, value in features_down.items():
        print(f"{key}: {value}")
    
    # Проверяем корректность основных признаков
    print("\nПроверка корректности признаков:")
    
    # Скорость изменения цены
    if features_up["price_velocity_5"] < 0 and features_down["price_velocity_5"] > 0:
        print("✓ Признак velocity правильно отражает направление тренда")
    else:
        print(f"✗ Признак velocity неверно отражает направление тренда")
    
    # Проверяем наличие признаков шума
    if "noise_level" in features_up and "noise_level" in features_down:
        print("✓ Признак noise_level успешно вычислен")
    else:
        print("✗ Признак noise_level отсутствует")
    
    # Проверяем наличие признаков торговой сессии
    session_keys = [key for key in features_up.keys() if key.startswith("session_")]
    if session_keys:
        print(f"✓ Признаки сессии успешно вычислены: {', '.join(session_keys)}")
    else:
        print("✗ Признаки сессии отсутствуют")
    
    # Проверяем кэширование
    print("\nПроверка кэширования признаков...")
    start_time = time.time()
    _ = feature_extractor.calculate_advanced_features(candles_up)
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    _ = feature_extractor.calculate_advanced_features(candles_up)
    second_call_time = time.time() - start_time
    
    if second_call_time < first_call_time:
        print(f"✓ Кэширование работает. Первый вызов: {first_call_time:.5f}с, второй вызов: {second_call_time:.5f}с")
    else:
        print(f"✗ Кэширование не работает. Первый вызов: {first_call_time:.5f}с, второй вызов: {second_call_time:.5f}с")

def test_enhanced_integration():
    """Тестирование усовершенствованного модуля интеграции ML"""
    print("\n--- Тестирование усовершенствованного модуля интеграции ML ---")
    
    # Создаем экземпляр усовершенствованного модуля интеграции
    integration = EnhancedMLIntegration()
    
    # Генерируем синтетические данные
    print("Генерация синтетических данных...")
    candles_up = generate_synthetic_candles(num_candles=50, trend="up")
    
    # Задаем "традиционные" результаты (как будто они получены от обычного алгоритма)
    conventional_results = ("up", 0.7, {"signal_sources": "Двойное дно (0.70)"})
    
    # Получаем прогноз от усовершенствованного модуля
    print("Получение прогноза от усовершенствованного модуля...")
    direction, confidence, info, is_enhanced = integration.analyze_pattern(
        candles_up, 
        conventional_results,
        currency_pair="EUR/USD"
    )
    
    # Выводим результаты
    print(f"\nРезультаты прогноза:")
    print(f"Направление: {direction}")
    print(f"Уверенность: {confidence:.2f}")
    print(f"Использовано улучшение: {is_enhanced}")
    
    print("\nДополнительная информация:")
    for key, value in info.items():
        if not isinstance(value, dict):  # Не выводим вложенные словари
            print(f"{key}: {value}")
    
    # Проверяем корректность базовых результатов
    if direction == "up":
        print("✓ Правильное направление прогноза")
    else:
        print(f"✗ Неправильное направление прогноза. Ожидалось: up, получено: {direction}")
    
    if confidence >= 0.7:
        print("✓ Уверенность на уровне или выше традиционного алгоритма")
    else:
        print(f"✗ Уверенность ниже традиционного алгоритма. Ожидалось: ≥0.7, получено: {confidence:.2f}")

def test_visualization():
    """Тестирование модуля визуализации"""
    print("\n--- Тестирование модуля визуализации ---")
    
    # Создаем экземпляр визуализатора
    visualizer = MLVisualizer(save_dir="test_visualizations")
    
    # Создаем тестовые данные для истории прогнозов
    print("Создание тестовых данных для истории прогнозов...")
    
    # Генерируем историю прогнозов для разных алгоритмов
    history = []
    for i in range(30):
        # Добавляем немного случайности в уверенность каждого алгоритма
        np.random.seed(i)
        
        # Для четных i делаем правильный прогноз, для нечетных - неправильный
        is_correct = (i % 2 == 0)
        
        # Базовая уверенность с небольшой случайностью
        conv_conf = 0.7 + np.random.normal(0, 0.1)
        ml_conf = 0.75 + np.random.normal(0, 0.1)
        lstm_conf = 0.8 + np.random.normal(0, 0.1)
        
        # Для неправильных прогнозов снижаем уверенность
        if not is_correct:
            conv_conf *= 0.9
            ml_conf *= 0.9
            lstm_conf *= 0.9
        
        # Объединенная уверенность
        combined_conf = 0.5 * conv_conf + 0.25 * ml_conf + 0.25 * lstm_conf
        
        # Создаем запись о прогнозе
        prediction = {
            'timestamp': datetime.now() - timedelta(minutes=i*5),
            'pair': 'EUR/USD',
            'direction': 'up',
            'confidence': combined_conf,
            'conventional_confidence': conv_conf,
            'ml_confidence': ml_conf,
            'lstm_confidence': lstm_conf,
            'result': is_correct,
            'noise_level': 0.01 + np.random.uniform(0, 0.05),
            'market_session': 'europe' if i % 3 == 0 else ('asia' if i % 3 == 1 else 'us'),
            'session_europe': 1 if i % 3 == 0 else 0,
            'session_asia': 1 if i % 3 == 1 else 0,
            'session_us': 1 if i % 3 == 2 else 0,
            'price_moved_up': is_correct  # Предполагаем, что предсказывали рост
        }
        
        history.append(prediction)
    
    # Добавляем историю прогнозов в визуализатор
    for pred in history:
        visualizer.add_prediction_to_history(pred)
    
    print(f"Добавлено {len(history)} прогнозов в историю")
    
    # Сохраняем историю прогнозов
    visualizer.save_predictions_history()
    
    # Создаем тестовые графики
    print("Создание тестовых графиков...")
    
    # График сравнения уверенности
    print("Создание графика сравнения уверенности...")
    confidence_fig = visualizer.plot_confidence_comparison(save=True, show=False)
    
    # График точности алгоритмов
    print("Создание графика точности алгоритмов...")
    accuracy_fig = visualizer.plot_algorithm_accuracy(save=True, show=False)
    
    # График производительности по торговым сессиям
    print("Создание графика производительности по торговым сессиям...")
    session_fig = visualizer.plot_market_session_performance(save=True, show=False)
    
    # График влияния шума
    print("Создание графика влияния шума...")
    noise_fig = visualizer.plot_noise_level_impact(save=True, show=False)
    
    # Генерация комплексного отчета
    print("Генерация комплексного отчета...")
    report_path = visualizer.generate_comprehensive_report()
    
    if report_path:
        print(f"✓ Отчет успешно сгенерирован: {report_path}")
    else:
        print("✗ Ошибка при генерации отчета")

def main():
    """Основная функция для запуска тестов"""
    print("===== Тестирование усовершенствованных ML-компонентов ScalpMaster =====")
    
    # Тестирование LSTM-анализатора
    test_lstm_analyzer()
    
    # Тестирование модуля расширенных признаков
    test_advanced_features()
    
    # Тестирование усовершенствованного модуля интеграции
    test_enhanced_integration()
    
    # Тестирование модуля визуализации
    test_visualization()
    
    print("\n===== Тестирование завершено =====")