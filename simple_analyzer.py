"""
Основной модуль анализатора паттернов на графиках валютных пар с расширенным логированием.
"""

import time
import cv2
import os
import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import threading
import logging
import json

# Импорты из новой структуры модулей
from screen_capture import capture_screen, select_monitor_area, take_screenshot
from candle_detection import detect_candles, preprocess_candles, find_local_extrema, analyze_trend
from pattern_detection import detect_patterns, detect_double_bottom, detect_double_top, detect_ascending_wedge, detect_descending_wedge, detect_trendline, draw_trendline
from visualization import display_analysis
from pattern_knowledge import PATTERN_KNOWLEDGE
from pattern_logger import PatternLogger  # Импорт нового модуля логирования

def colored_print(text, color=None, style=None, add_border=False, border_width=60):
    """
    Функция для вывода цветного текста в консоль без использования внешних библиотек.
    
    Args:
        text: текст для вывода
        color: цвет текста ('red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white')
        style: стиль текста ('bold', 'underline', 'blink', 'reverse')
        add_border: добавить ли рамку вокруг текста
        border_width: ширина рамки
    """
    # ANSI коды цветов для текста
    colors = {
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
        'white': '37'
    }
    
    # ANSI коды для стилей
    styles = {
        'bold': '1',
        'underline': '4',
        'blink': '5',
        'reverse': '7'
    }
    
    # Формируем код форматирования
    format_code = []
    if color and color in colors:
        format_code.append(colors[color])
    if style and style in styles:
        format_code.append(styles[style])
    
    if add_border:
        border = "=" * border_width
        print(border)
    
    if format_code:
        # Применяем форматирование
        formatted_text = f"\033[{';'.join(format_code)}m{text}\033[0m"
        print(formatted_text)
    else:
        # Если нет форматирования, просто выводим текст
        print(text)
    
    if add_border:
        border = "=" * border_width
        print(border)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler("sm_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("simple_analyzer")

class SimplePatternAnalyzer:
    def __init__(self):
        # Инициализируем атрибуты класса
        self.pattern_names = ["Двойное дно", "Двойная вершина", "Восходящий клин", "Нисходящий клин", "Линия тренда"]
        self.running = False
        self.monitor_area = None
        self.current_candles = []
        self.current_pattern = None
        self.pattern_confidence = 0
        self.last_notification_time = time.time() - 300  # Начальное значение для избежания спама
        self.min_confidence_threshold = 0.75  # Минимальный порог уверенности для оповещения
        self.patience_level = 3  # Уровень "терпеливости" системы (1-5)
        self.pattern_knowledge = PATTERN_KNOWLEDGE
        self.paused = False  # Флаг приостановки выдачи новых прогнозов
        self.trendline_info = {}  # Информация о линии тренда для отрисовки
        
        # Настройки временных интервалов для разных типов паттернов
        self.pattern_mode = "all"  # По умолчанию анализируем все паттерны
        self.pattern_settings = {
            "reversal": {
                "candle_time": 1,    # 1 минута для свечи
                "forecast_time": 3,  # 3 минуты для прогноза
                "patterns": ["Двойное дно", "Двойная вершина"]
            },
            "wedge": {
                "candle_time": 5,     # 5 минут для свечи
                "forecast_time": 15,  # 15 минут для прогноза (соотношение 1:3)
                "patterns": ["Восходящий клин", "Нисходящий клин"]
            },
            "trendline": {
                "candle_time": 1,     # 1 минута для свечи
                "forecast_time": 3,   # 3 минуты для прогноза
                "patterns": ["Линия тренда"]
            },
            "all": {
                "candle_time": 1,    # 1 минута для свечи
                "forecast_time": 3,  # 3 минуты для прогноза (как для разворотных)
                "patterns": ["Двойное дно", "Двойная вершина", "Восходящий клин", "Нисходящий клин", "Линия тренда"]
            }
        }
        
        # Текущие настройки (по умолчанию - для всех паттернов)
        self.candle_time = self.pattern_settings["all"]["candle_time"] * 60  # в секундах
        self.forecast_time = self.pattern_settings["all"]["forecast_time"]   # в минутах
        self.enabled_patterns = self.pattern_settings["all"]["patterns"]     # список активных паттернов
        
        # Новые атрибуты для отслеживания прогнозов
        self.forecasts = []  # Список активных прогнозов
        self.completed_forecasts = []  # История проверенных прогнозов
        self.forecast_log_file = "forecast_log.csv"  # Файл для сохранения прогнозов
        self.stats = {
            "Двойное дно": {"correct": 0, "incorrect": 0},
            "Двойная вершина": {"correct": 0, "incorrect": 0},
            "Восходящий клин": {"correct": 0, "incorrect": 0},
            "Нисходящий клин": {"correct": 0, "incorrect": 0},
            "Линия тренда": {"correct": 0, "incorrect": 0},
            "all": {"correct": 0, "incorrect": 0}
        }
        
        # Добавляем настройки порогов для разных типов паттернов
        self.pattern_thresholds = {
            "Двойное дно": 0.2,
            "Двойная вершина": 0.2,
            "Восходящий клин": 0.25,
            "Нисходящий клин": 0.25,
            "Линия тренда": 0.2
        }
        
        # Инициализация лог-файла, если его нет
        self._init_forecast_log()

        # Инициализация расширенного логгера паттернов
        self.pattern_logger = PatternLogger("pattern_forecast_log.csv")
    
    def set_pattern_mode(self, mode):
        """Установка режима анализа паттернов"""
        if mode in self.pattern_settings:
            self.pattern_mode = mode
            settings = self.pattern_settings[mode]
            self.candle_time = settings["candle_time"] * 60  # в секундах
            self.forecast_time = settings["forecast_time"]   # в минутах
            self.enabled_patterns = settings["patterns"]     # список активных паттернов
            
            logger.info(f"Установлен режим анализа: {mode}")
            logger.info(f"Свеча: {settings['candle_time']} мин, прогноз: {settings['forecast_time']} мин")
            logger.info(f"Активные паттерны: {', '.join(self.enabled_patterns)}")
        else:
            logger.warning(f"Ошибка: неизвестный режим анализа '{mode}'")

    def toggle_pattern(self, pattern_name, enabled=True):
        """Включение или отключение конкретного паттерна"""
        if pattern_name in self.pattern_names:
            if enabled and pattern_name not in self.enabled_patterns:
                self.enabled_patterns.append(pattern_name)
                logger.info(f"Паттерн '{pattern_name}' включен")
            elif not enabled and pattern_name in self.enabled_patterns:
                self.enabled_patterns.remove(pattern_name)
                logger.info(f"Паттерн '{pattern_name}' отключен")
            else:
                logger.info(f"Паттерн '{pattern_name}' уже {'включен' if enabled else 'отключен'}")
        else:
            logger.warning(f"Паттерн '{pattern_name}' не найден")

    def _init_forecast_log(self):
        """Инициализация файла логирования прогнозов"""
        try:
            if not os.path.exists(self.forecast_log_file):
                logger.info(f"Создание нового лог-файла прогнозов: {self.forecast_log_file}")
                with open(self.forecast_log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Время прогноза', 'Паттерн', 'Уверенность', 'Направление', 
                                    'Предсказание', 'Время проверки', 'Результат', 'Правильно'])
            else:
                logger.info(f"Лог-файл прогнозов существует: {self.forecast_log_file}")
                
                # Проверяем наличие данных и заголовка
                with open(self.forecast_log_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                        # Проверяем, соответствует ли заголовок ожидаемому формату
                        if len(header) < 8 or header[0] != 'Время прогноза':
                            logger.warning("Заголовок лог-файла имеет неверный формат")
                            logger.warning(f"Ожидаемый формат: ['Время прогноза', 'Паттерн', 'Уверенность', ...]")
                            logger.warning(f"Текущий формат: {header}")
                    except StopIteration:
                        logger.warning("Лог-файл пуст или поврежден, создаем заголовок")
                        with open(self.forecast_log_file, 'w', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(['Время прогноза', 'Паттерн', 'Уверенность', 'Направление', 
                                            'Предсказание', 'Время проверки', 'Результат', 'Правильно'])
        except Exception as e:
            logger.error(f"Ошибка при инициализации лог-файла: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        # Загрузка существующей статистики
        self._load_forecast_stats()
           
    def _load_forecast_stats(self):
        """Загрузка статистики из лог-файла"""
        try:
            with open(self.forecast_log_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Пропускаем заголовок
                
                # Сбрасываем статистику
                for pattern in self.stats:
                    self.stats[pattern] = {"correct": 0, "incorrect": 0}
                
                # Читаем данные
                for row in reader:
                    if len(row) >= 8:  # Проверка полноты данных
                        pattern = row[1]
                        is_correct = row[7] == 'True'
                        
                        if pattern in self.stats:
                            if is_correct:
                                self.stats[pattern]["correct"] += 1
                                self.stats["all"]["correct"] += 1
                            else:
                                self.stats[pattern]["incorrect"] += 1
                                self.stats["all"]["incorrect"] += 1
                                
            logger.info("Статистика прогнозов загружена успешно")
        except Exception as e:
            logger.error(f"Ошибка при загрузке статистики: {e}")
            import traceback
            logger.error(traceback.format_exc())

def _extract_key_points_from_pattern(self, pattern, candles):
    """
    Извлечение ключевых точек паттерна для логирования
    
    Args:
        pattern: название паттерна
        candles: список свечей
        
    Returns:
        dict: словарь с ключевыми точками паттерна
    """
    key_points = {}
    
    if len(candles) < 3:
        return key_points
    
    try:
        if pattern == "Двойное дно":
            # Находим два минимума
            local_mins = []
            for i in range(1, len(candles) - 1):
                if candles[i]["bottom"] > candles[i-1]["bottom"] and candles[i]["bottom"] > candles[i+1]["bottom"]:
                    local_mins.append((i, candles[i]["bottom"]))
            
            if len(local_mins) >= 2:
                key_points["min1_idx"] = local_mins[-2][0]
                key_points["min1_y"] = local_mins[-2][1]
                key_points["min2_idx"] = local_mins[-1][0]
                key_points["min2_y"] = local_mins[-1][1]
                
        elif pattern == "Двойная вершина":
            # Находим два максимума
            local_maxs = []
            for i in range(1, len(candles) - 1):
                if candles[i]["top"] < candles[i-1]["top"] and candles[i]["top"] < candles[i+1]["top"]:
                    local_maxs.append((i, candles[i]["top"]))
            
            if len(local_maxs) >= 2:
                key_points["max1_idx"] = local_maxs[-2][0]
                key_points["max1_y"] = local_maxs[-2][1]
                key_points["max2_idx"] = local_maxs[-1][0]
                key_points["max2_y"] = local_maxs[-1][1]
                
        elif pattern == "Восходящий клин" or pattern == "Нисходящий клин":
            # Используем информацию о линии тренда
            key_points["trend_start_x"] = candles[0]["center_x"]
            key_points["trend_start_y"] = candles[0]["center_y"]
            key_points["trend_end_x"] = candles[-1]["center_x"]
            key_points["trend_end_y"] = candles[-1]["center_y"]
            
        elif pattern == "Линия тренда" and self.trendline_info:
            # Используем сохраненную информацию о линии тренда
            for key, value in self.trendline_info.items():
                if isinstance(value, (int, float, str, bool)):
                    key_points[key] = value
    
    except Exception as e:
        logger.error(f"Ошибка при извлечении ключевых точек: {e}")
    
    return key_points

def _add_forecast(self, pattern, confidence, direction, message):
    """
    Добавление нового прогноза для отслеживания с расширенным логированием
    """
    # Отладочная информация
    logger.info(f"Добавление прогноза: {pattern}, уверенность: {confidence:.2f}, направление: {direction}")
    
    # Текущее время
    current_time = datetime.now()
    
    # Время начала прогноза (текущее время + 2 секунды)
    forecast_time = current_time + timedelta(seconds=2)
    
    # Время проверки (время начала прогноза + время прогноза из настроек)
    check_time = forecast_time + timedelta(minutes=self.forecast_time)
    
    # Захват текущего изображения для определения начальной позиции
    frame = capture_screen(self.monitor_area)
    detected_candles = detect_candles(frame)
    
    # Создаем запись о прогнозе
    forecast = {
        "time": forecast_time,      # Время начала прогноза (через 2 секунды)
        "pattern": pattern,
        "confidence": confidence,
        "direction": direction,
        "message": message,
        "check_time": check_time,   # Время проверки (начало + время прогноза)
        "result": None,             # Будет заполнено при проверке
        "is_correct": None,         # Будет заполнено при проверке
        # Дополнительные поля для расширенного логирования
        "timeframe": self.candle_time // 60,  # Таймфрейм в минутах
        "expiration": self.forecast_time,     # Время экспирации в минутах
        "currency_pair": "",        # В SimplePatternAnalyzer не отслеживается валютная пара
    }
    
    # Сохраняем начальную цену, если есть свечи
    if detected_candles:
        forecast["initial_price"] = detected_candles[-1]["center_y"]
        logger.info(f"Начальная цена: {forecast['initial_price']}")
        
        # Извлекаем ключевые точки паттерна для логирования
        forecast["key_points"] = self._extract_key_points_from_pattern(pattern, detected_candles)
    else:
        logger.warning("Не удалось определить начальную цену")
    
    # Добавляем в список активных прогнозов
    self.forecasts.append(forecast)
    
    # Сохраняем прогноз в расширенный лог (без результата)
    self.pattern_logger.log_pattern_forecast(forecast)
    
    # Выводим информацию о прогнозе с учетом измененного времени
    logger.info(f"Начало прогноза: {forecast_time.strftime('%H:%M:%S')} (через 2 сек)")
    logger.info(f"Проверка прогноза: {check_time.strftime('%H:%M:%S')} (через {self.forecast_time} мин)")
    
    # Вычисляем задержку для проверки (время проверки - текущее время)
    check_delay = (check_time - current_time).total_seconds()
    
    # Запускаем функцию запланированной проверки в отдельном потоке
    check_thread = threading.Timer(check_delay, self._check_forecast, args=[forecast])
    check_thread.daemon = True  # Чтобы поток не блокировал завершение программы
    check_thread.start()

    def _check_forecast(self, forecast):
        """
        Проверка результата прогноза с цветным форматированием без дублирования
        """
        if not self.running:
            return  # Если программа остановлена, не проверяем
        
        # Логируем проверку только в файл, не в консоль
        file_handler = next((h for h in logger.handlers if isinstance(h, logging.FileHandler)), None)
        if file_handler:
            log_message = f"Проверка прогноза от {forecast['time'].strftime('%H:%M:%S')}..."
            file_handler.emit(logging.LogRecord(
                name=logger.name, 
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg=log_message,
                args=(),
                exc_info=None
            ))
        
        # Захват текущего изображения для проверки
        frame = capture_screen(self.monitor_area)
        
        # Обнаружение свечей
        detected_candles = detect_candles(frame)
        
        # Определение фактического движения цены
        if not detected_candles:
            # Логируем ошибку только в файл
            if file_handler:
                log_message = "Не удалось обнаружить свечи для проверки прогноза"
                file_handler.emit(logging.LogRecord(
                    name=logger.name, 
                    level=logging.WARNING,
                    pathname='',
                    lineno=0,
                    msg=log_message,
                    args=(),
                    exc_info=None
                ))
            return
        
        # Используем самую правую (новую) свечу
        current_price = detected_candles[-1]["center_y"]
        
        # Сохраняем конечную цену для логирования
        forecast["final_price"] = current_price
        
        # Проверяем правильность прогноза по сравнению с исходной ценой
        predicted_up = "верх" in forecast["direction"].lower()
        
        # Используем исходную цену для сравнения, если она доступна
        if "initial_price" in forecast:
            # Логируем цены только в файл
            if file_handler:
                log_message = f"Начальная цена: {forecast['initial_price']}, Текущая цена: {current_price}"
                file_handler.emit(logging.LogRecord(
                    name=logger.name, 
                    level=logging.INFO,
                    pathname='',
                    lineno=0,
                    msg=log_message,
                    args=(),
                    exc_info=None
                ))
            actual_up = current_price < forecast["initial_price"]  # На экране Y растет вниз
        else:
            # Резервный вариант, если исходная цена не была сохранена
            if file_handler:
                log_message = "Начальная цена не была сохранена"
                file_handler.emit(logging.LogRecord(
                    name=logger.name, 
                    level=logging.WARNING,
                    pathname='',
                    lineno=0,
                    msg=log_message,
                    args=(),
                    exc_info=None
                ))
            screen_center = self.monitor_area["height"] // 2
            actual_up = current_price < screen_center
        
        # Определяем, был ли прогноз верным
        is_correct = (predicted_up == actual_up)
        
        # Обновляем информацию о прогнозе
        forecast["result"] = "Вверх" if actual_up else "Вниз"
        forecast["is_correct"] = is_correct
        forecast["check_time"] = datetime.now()  # Обновляем время проверки
        
        # Обновляем статистику
        pattern = forecast["pattern"]
        if pattern not in self.stats:
            # Если паттерн новый, добавляем его в статистику
            self.stats[pattern] = {"correct": 0, "incorrect": 0}
            
        if is_correct:
            self.stats[pattern]["correct"] += 1
            self.stats["all"]["correct"] += 1
        else:
            self.stats[pattern]["incorrect"] += 1
            self.stats["all"]["incorrect"] += 1
        
        # Добавляем в историю и удаляем из активных прогнозов
        self.completed_forecasts.append(forecast)
        if forecast in self.forecasts:
            self.forecasts.remove(forecast)
        
        # Логируем результат в старый лог (только запись в файл)
        if file_handler:
            log_message = f"Запись результата в лог: {forecast['pattern']}, {forecast['result']}, Верно: {is_correct}"
            file_handler.emit(logging.LogRecord(
                name=logger.name, 
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg=log_message,
                args=(),
                exc_info=None
            ))
        
        self._log_forecast_result(forecast)
        
        # Логируем результат в новый расширенный лог
        self.pattern_logger.update_forecast_result(
            forecast["time"],
            {
                "result": forecast["result"],
                "is_correct": forecast["is_correct"],
                "final_price": forecast["final_price"],
                "check_time": forecast["check_time"]
            }
        )
        
        # Выводим результат на экран с цветным форматированием
        result_text = "ВЕРНО" if is_correct else "НЕВЕРНО"
        result_color = "green" if is_correct else "red"
        
        colored_print("", add_border=True)
        colored_print("РЕЗУЛЬТАТ ПРОГНОЗА:", color="blue", style="bold")
        colored_print(f"Паттерн: {forecast['pattern']} ({forecast['confidence']:.2f})", color="cyan")
        colored_print(f"Прогноз: {forecast['direction']}", color="cyan")
        colored_print(f"Фактическое движение: {forecast['result']}", color="cyan")
        colored_print(f"Оценка: {result_text}", color=result_color, style="bold")
        colored_print(f"Время прогноза: {forecast['time'].strftime('%H:%M:%S')}", color="magenta")
        colored_print(f"Время проверки: {datetime.now().strftime('%H:%M:%S')}", color="magenta")
        colored_print("", add_border=True)
        
        # Показать обновленную статистику только в файл
        if file_handler:
            self._show_forecast_stats(to_console=False)

    def _log_forecast_result(self, forecast):
        """Запись результата прогноза в CSV-файл (оригинальный метод, оставлен для совместимости)"""
        try:
            with open(self.forecast_log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row = [
                    forecast["time"].strftime('%Y-%m-%d %H:%M:%S'),
                    forecast["pattern"],
                    f"{forecast['confidence']:.2f}",
                    forecast["direction"],
                    forecast["message"],
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    forecast["result"],
                    str(forecast["is_correct"])
                ]
                writer.writerow(row)
                logger.info(f"Запись в лог успешно добавлена")
        except Exception as e:
            logger.error(f"Ошибка при записи в лог-файл: {e}")
            import traceback
            logger.error(traceback.format_exc())

def _show_forecast_stats(self, to_console=True):
    """Отображение статистики прогнозов"""
    stats_info = []
    stats_info.append("Статистика прогнозов:")
    stats_info.append("-" * 60)
    stats_info.append(f"{'Паттерн':<20} {'Верных':<10} {'Неверных':<10} {'Точность':<10}")
    stats_info.append("-" * 60)
    
    for pattern in self.stats:
        correct = self.stats[pattern]["correct"]
        incorrect = self.stats[pattern]["incorrect"]
        total = correct + incorrect
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        stats_info.append(f"{pattern:<20} {correct:<10} {incorrect:<10} {accuracy:.1f}%")
    
    stats_info.append("-" * 60)
    
    # Вывод в лог и консоль или только в лог
    if to_console:
        for line in stats_info:
            if pattern == "all":
                colored_print(line, color="green", style="bold")
            else:
                colored_print(line, color="cyan")
    else:
        # Только в лог-файл
        file_handler = next((h for h in logger.handlers if isinstance(h, logging.FileHandler)), None)
        if file_handler:
            log_message = "\n".join(stats_info)
            file_handler.emit(logging.LogRecord(
                name=logger.name, 
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg=log_message,
                args=(),
                exc_info=None
            ))

    def _notify_user(self, pattern, confidence, direction, message, show_details=False):
        """Оповещение пользователя о значимом паттерне с цветным форматированием без дублирования"""
        # Если выдача прогнозов приостановлена, просто выходим без вывода сообщений
        if self.paused:
            return
        
        current_time = time.time()
        
        # Проверка, прошло ли достаточно времени с последнего оповещения
        if current_time - self.last_notification_time < 60:  # Минимум 60 секунд между оповещениями
            return
        
        self.last_notification_time = current_time
        
        # Добавляем прогноз для отслеживания с расширенным логированием
        self._add_forecast(pattern, confidence, direction, message)
        
        # Воспроизведение звукового сигнала
        self._play_sound_notification()
        
        # Вывод информации в консоль с цветом и без дублирования в лог
        colored_print("", add_border=True)
        colored_print(f"СИГНАЛ: {pattern} ({confidence:.2f})", color="yellow", style="bold")
        colored_print(f"НАПРАВЛЕНИЕ: {direction}", color="cyan", style="bold")
        colored_print(f"ПРОГНОЗ: {message}", color="green")
        colored_print(f"ВРЕМЯ: {datetime.now().strftime('%H:%M:%S')}", color="magenta")
        colored_print("", add_border=True)
        
        # Логируем в файл без вывода в консоль через логер
        file_handler = next((h for h in logger.handlers if isinstance(h, logging.FileHandler)), None)
        if file_handler:
            log_message = f"СИГНАЛ: {pattern} ({confidence:.2f})\nНАПРАВЛЕНИЕ: {direction}\nПРОГНОЗ: {message}\nВРЕМЯ: {datetime.now().strftime('%H:%M:%S')}"
            file_handler.emit(logging.LogRecord(
                name=logger.name, 
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg=log_message,
                args=(),
                exc_info=None
            ))
        
        # Добавляем задержку после сообщения о прогнозе
        time.sleep(2)  # Пауза на 2 секунды