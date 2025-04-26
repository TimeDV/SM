"""
Анализатор бинарных опционов для ScalpMaster (SM)
Расширяет функциональность SimplePatternAnalyzer для работы с бинарными опционами
"""

import time
import cv2
import os
import csv
import numpy as np
from datetime import datetime, timedelta
import threading
import winsound
import logging
import sys

from simple_analyzer import SimplePatternAnalyzer
from screen_capture import capture_screen, select_monitor_area, take_screenshot
from candle_detection import detect_candles
from pattern_detection import detect_patterns, detect_trendline_signal
from visualization import display_binary_analysis
from currency_profiles import CURRENCY_PROFILES
from binary_indicators import (
    calculate_rsi, detect_bollinger_touch, 
    detect_momentum_signal, detect_pin_bar,
    detect_engulfing_pattern, detect_false_breakout
)

# Настройка логирования с принудительной установкой UTF-8
logger = logging.getLogger("binary_analyzer")

# Исправляем проблему кодировки для вывода в консоль
# Создаем специальный обработчик для консоли, который корректно обрабатывает Unicode
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')  # Убираем лишнюю информацию
console_handler.setFormatter(console_formatter)

# Удаляем существующие обработчики и добавляем новый
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Добавляем обработчик для файла
file_handler = logging.FileHandler("sm_binary.log", encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def play_wav(file_path):
    """
    Воспроизводит WAV файл, используя стандартную библиотеку
    
    Args:
        file_path: путь к WAV файлу
    """
    try:
        # Асинхронное воспроизведение (SND_ASYNC)
        winsound.PlaySound(file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        logger.error(f"Ошибка при воспроизведении WAV: {e}")

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

class BinaryPatternAnalyzer(SimplePatternAnalyzer):
    def __init__(self):
        super().__init__()
        self.current_pair = "EUR/USD"  # По умолчанию
        self.timeframe = 1  # Минуты
        self.expiration_time = 3  # Минуты
        
        # Настройки для бинарных опционов
        self.signal_strength = 0  # Сила текущего сигнала (0-100)
        self.signal_direction = None  # "up" или "down"
        self.use_composite_signals = True  # Использовать комбинацию индикаторов
        self.signal_sources = ""  # Источники сигналов
        
        # Веса для разных источников сигналов
        self.signal_weights = {
            "patterns": 0.4,  # Свечные паттерны
            "indicators": 0.4,  # Технические индикаторы
            "momentum": 0.2    # Моментум
        }
        
        # Поля для бинарных опционов
        self.suggested_trade_amount = 0  # Рекомендуемая сумма ставки
        self.binary_log_file = "binary_signals_log.csv"  # Файл для логирования сигналов
        
        # Инициализация лога и настроек
        self._init_binary_log()
        self._update_thresholds_from_profile()
        
        # Поля для отслеживания результатов прогнозов
        self.prediction_starting_price = 0
        self.prediction_results = []  # История результатов
    
    def _init_binary_log(self):
        """Инициализация файла логирования бинарных сигналов"""
        if not os.path.exists(self.binary_log_file):
            with open(self.binary_log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Время', 'Валютная пара', 'Таймфрейм', 'Экспирация', 
                                 'Сигнал', 'Уверенность', 'Источники', 'Результат', 'P&L'])
            logger.info(f"Создан новый лог-файл бинарных сигналов: {self.binary_log_file}")
    
    def _update_thresholds_from_profile(self):
        """Обновление порогов чувствительности из профиля валютной пары"""
        if self.current_pair not in CURRENCY_PROFILES:
            return
            
        pattern_map = {
            "double_bottom": "Двойное дно",
            "double_top": "Двойная вершина",
            "ascending_wedge": "Восходящий клин",
            "descending_wedge": "Нисходящий клин"
        }
        
        profile = CURRENCY_PROFILES[self.current_pair]
        for pattern_key, threshold in profile["pattern_sensitivity"].items():
            if pattern_key in pattern_map and pattern_map[pattern_key] in self.pattern_thresholds:
                self.pattern_thresholds[pattern_map[pattern_key]] = threshold
    
    def set_currency_pair(self, pair):
        """Установка текущей валютной пары"""
        if pair in CURRENCY_PROFILES:
            self.current_pair = pair
            self._update_thresholds_from_profile()
            logger.info(f"Валютная пара изменена на {pair} с настроенными порогами чувствительности")
            return True
        else:
            logger.warning(f"Ошибка: профиль для пары {pair} не найден")
            return False
    
    def set_timeframe(self, minutes):
        """Установка таймфрейма для анализа"""
        self.timeframe = minutes
        # Настраиваем время свечи в секундах
        self.candle_time = minutes * 60
        logger.info(f"Таймфрейм установлен: {minutes} мин")
    
    def set_expiration_time(self, minutes):
        """Установка времени экспирации для прогноза"""
        self.expiration_time = minutes
        # Настраиваем время прогноза
        self.forecast_time = minutes
        logger.info(f"Время экспирации установлено: {minutes} мин")
        
    def pause_forecasts(self):
        """Приостановка выдачи прогнозов"""
        self.paused = True
        logger.info("Выдача прогнозов приостановлена")
        print("Выдача прогнозов приостановлена")

    def resume_forecasts(self):
        """Возобновление выдачи прогнозов"""
        self.paused = False
        logger.info("Выдача прогнозов возобновлена")
        print("Выдача прогнозов возобновлена")
        
    def play_message_sound(self):
        """Воспроизведение звука сообщения"""
        try:
            play_wav("message.wav")
        except Exception as e:
            logger.error(f"Ошибка при воспроизведении звука сообщения: {e}")
    
    def analyze_binary_signal(self, candles):
        """
        Комплексный анализ для бинарных опционов с использованием нескольких индикаторов
        
        Args:
            candles: список обнаруженных свечей
            
        Returns:
            tuple: (направление сигнала, уверенность, использовался ли ML)
        """
        if len(candles) < 10:
            return None, 0.0, False  # Недостаточно данных
        
        signals = []
        confidence_sum = 0
        signal_sources = []
        
        # 1. Анализ классических паттернов
        # Заменяем вызов _detect_patterns на прямое использование функции из pattern_detection
        from pattern_detection import detect_patterns
        pattern, pattern_conf, pattern_info = detect_patterns(candles, self.pattern_thresholds, self.enabled_patterns)
        
        if pattern and pattern_conf >= self.min_confidence_threshold:
            # Определяем направление на основе типа паттерна
            direction = "up" if pattern in ["Двойное дно", "Нисходящий клин"] else "down"
            signals.append((direction, pattern_conf, "patterns"))
            confidence_sum += pattern_conf * self.signal_weights["patterns"]
            signal_sources.append(f"{pattern} ({pattern_conf:.2f})")
        
        # 2. Анализ RSI
        rsi_value = calculate_rsi(candles)
        if self.current_pair in CURRENCY_PROFILES:
            rsi_threshold = CURRENCY_PROFILES[self.current_pair]["rsi_levels"]
            
            if rsi_value <= rsi_threshold["oversold"]:
                # Перепроданность - сигнал на рост
                rsi_conf = 0.5 + (rsi_threshold["oversold"] - rsi_value) / 60
                rsi_conf = min(rsi_conf, 0.95)
                signals.append(("up", rsi_conf, "indicators"))
                confidence_sum += rsi_conf * self.signal_weights["indicators"]
                signal_sources.append(f"RSI перепродан ({rsi_value:.1f})")
                
            elif rsi_value >= rsi_threshold["overbought"]:
                # Перекупленность - сигнал на падение
                rsi_conf = 0.5 + (rsi_value - rsi_threshold["overbought"]) / 60
                rsi_conf = min(rsi_conf, 0.95)
                signals.append(("down", rsi_conf, "indicators"))
                confidence_sum += rsi_conf * self.signal_weights["indicators"]
                signal_sources.append(f"RSI перекуплен ({rsi_value:.1f})")
        
        # 3. Анализ полос Боллинджера
        bb_signal = detect_bollinger_touch(candles)
        if bb_signal:
            direction = "down" if bb_signal == "upper" else "up"
            signals.append((direction, 0.8, "indicators"))
            confidence_sum += 0.8 * self.signal_weights["indicators"]
            signal_sources.append(f"Касание {'верхней' if bb_signal == 'upper' else 'нижней'} полосы Боллинджера")
        
        # 4. Анализ моментума
        mom_direction, mom_strength = detect_momentum_signal(candles)
        if mom_direction:
            mom_conf = mom_strength / 100
            signals.append((mom_direction, mom_conf, "momentum"))
            confidence_sum += mom_conf * self.signal_weights["momentum"]
            signal_sources.append(f"Моментум {mom_direction} ({mom_strength:.1f}%)")
        
        # 5. Анализ Пин-бара (сильный сигнал разворота)
        pin_direction, pin_conf = detect_pin_bar(candles)
        if pin_direction and pin_conf > 0.7:
            signals.append((pin_direction, pin_conf, "patterns"))
            confidence_sum += pin_conf * self.signal_weights["patterns"]
            signal_sources.append(f"Пин-бар {pin_direction} ({pin_conf:.2f})")
        
        # 6. Паттерн поглощения
        engulf_direction, engulf_conf = detect_engulfing_pattern(candles)
        if engulf_direction and engulf_conf > 0.7:
            signals.append((engulf_direction, engulf_conf, "patterns"))
            confidence_sum += engulf_conf * self.signal_weights["patterns"]
            signal_sources.append(f"Поглощение {engulf_direction} ({engulf_conf:.2f})")
        
        # 7. Ложный пробой
        fake_direction, fake_conf = detect_false_breakout(candles)
        if fake_direction and fake_conf > 0.7:
            signals.append((fake_direction, fake_conf, "patterns"))
            confidence_sum += fake_conf * self.signal_weights["patterns"]
            signal_sources.append(f"Ложный пробой {fake_direction} ({fake_conf:.2f})")
            
        # 8. Анализ линии тренда
        tl_direction, tl_conf = detect_trendline_signal(candles)
        if tl_direction and tl_conf > 0.6:
            signals.append((tl_direction, tl_conf, "patterns"))
            confidence_sum += tl_conf * self.signal_weights["patterns"]
            signal_sources.append(f"Линия тренда {tl_direction} ({tl_conf:.2f})")
        
        # Сохраняем источники сигналов
        self.signal_sources = ", ".join(signal_sources)
        
        # Принятие решения на основе всех сигналов
        if not signals:
            self.signal_direction = None
            self.signal_strength = 0
            self.signal_sources = ""
            return None, 0, False
            
        # Подсчитываем взвешенные значения сигналов по направлениям
        up_signals = sum(conf * self.signal_weights[src] for dir, conf, src in signals if dir == "up")
        down_signals = sum(conf * self.signal_weights[src] for dir, conf, src in signals if dir == "down")
        
        # Определяем финальное направление и силу сигнала
        if up_signals > down_signals and up_signals > 0.4:
            self.signal_direction = "up"
            self.signal_strength = min(up_signals * 100, 95)  # Сила сигнала от 0 до 95%
            return "up", up_signals
        elif down_signals > up_signals and down_signals > 0.4:
            self.signal_direction = "down"
            self.signal_strength = min(down_signals * 100, 95)  # Сила сигнала от 0 до 95%
            return "down", down_signals
        
        # Нет четкого сигнала
        self.signal_direction = None
        self.signal_strength = 0
        self.signal_sources = ""
        return None, 0    
    def get_recommended_action(self):
        """
        Получение рекомендации для действия на бинарной платформе
        
        Returns:
            tuple: (действие, сообщение, рекомендуемая ставка)
        """
        if not self.signal_direction or self.signal_strength < 50:
            return "HOLD", "Недостаточно сильный сигнал для входа", 0
        
        # Определяем уровень уверенности и риск
        if self.signal_strength >= 85:
            confidence_text = "ОЧЕНЬ СИЛЬНЫЙ СИГНАЛ"
            risk_factor = 0.05  # 5% от баланса
        elif self.signal_strength >= 70:
            confidence_text = "СИЛЬНЫЙ СИГНАЛ"
            risk_factor = 0.03  # 3% от баланса
        else:
            confidence_text = "СРЕДНИЙ СИГНАЛ"
            risk_factor = 0.01  # 1% от баланса
        
        # Предполагаемый баланс (можно заменить на реальное значение)
        balance = 30000  # Рублей
        
        # Рассчитываем рекомендуемую ставку
        suggested_amount = int(balance * risk_factor)
        self.suggested_trade_amount = suggested_amount
        
        # Форматируем сообщение
        start_time = self._get_future_time_str(0, 2)
        end_time = self._get_future_time_str(self.expiration_time, 2)
        
        # Используем слова вместо эмодзи для избежания проблем с кодировкой
        action_text = "ПОКУПКА" if self.signal_direction == "up" else "ПРОДАЖА"
        message = f"{action_text} {confidence_text} (с {start_time} до {end_time})"
        action = "UP" if self.signal_direction == "up" else "DOWN"
        
        return action, message, suggested_amount
    
    def _get_future_time_str(self, minutes=0, seconds=0):
        """
        Получение строки с временем, которое наступит через указанное количество минут и секунд
        
        Args:
            minutes: количество минут
            seconds: количество секунд
            
        Returns:
            str: строка с временем в формате HH:MM:SS
        """
        future_time = datetime.now() + timedelta(minutes=minutes, seconds=seconds)
        return future_time.strftime('%H:%M:%S')
    
    def _notify_binary_signal(self, action, confidence, message):
        """
        Оповещение о сигнале для бинарных опционов
        
        Args:
            action: действие (UP/DOWN)
            confidence: уверенность (0-100)
            message: сообщение для отображения
        """
        # Если выдача прогнозов приостановлена или недавно было оповещение, выходим
        if self.paused or time.time() - self.last_notification_time < 10:
            return
        
        self.last_notification_time = time.time()
        
        # Сохраняем начальную цену для последующей проверки прогноза
        if self.current_candles:
            self.prediction_starting_price = self.current_candles[-1]["center_y"]
            
            # Используем точное время для идентификации сигнала
            signal_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            direction = "up" if action == "UP" else "down"
            
            # Записываем прогноз в лог паттернов
            self.log_pattern_forecast(
                pattern=self.signal_sources if self.signal_sources else "Комплексный сигнал",
                confidence=confidence/100.0,
                direction=direction,
                initial_price=self.prediction_starting_price
            )
            
            # Запускаем таймер для проверки
            self._schedule_prediction_check(signal_time, direction, self.expiration_time)
        
        # Воспроизведение звукового сигнала в зависимости от уверенности
        direction = "up" if action == "UP" else "down"
        self._play_signal_sound(confidence, direction) 
        
        # Вывод информации в консоль с избеганием эмодзи
        self._print_signal_info(action, confidence, message)
        
        # Логируем сигнал
        self._log_binary_signal(action, confidence)
    
    def _print_signal_info(self, action, confidence, message):
        """
        Отображение информации о сигнале в консоли в более заметном формате
        с цветным текстом и без дублирования в логи
        """
        # Определяем цвет в зависимости от действия
        color = 'green' if action == "UP" else 'red'
        action_text = "ВВЕРХ" if action == "UP" else "ВНИЗ"
        
        # Формируем сообщение с необходимой информацией
        colored_print("", add_border=True)
        colored_print(f"!!! СИГНАЛ: {action_text} ({confidence:.1f}%) !!!", color=color, style='bold')
        colored_print(message, color='yellow')
        colored_print("")
        colored_print(f"ВРЕМЯ: {datetime.now().strftime('%H:%M:%S')}", color='cyan')
        colored_print(f"ПАРА: {self.current_pair} (TF: {self.timeframe}m, EXP: {self.expiration_time}m)", color='cyan')
        colored_print(f"ИСТОЧНИКИ: {self.signal_sources}", color='cyan')
        colored_print("")
        colored_print(f"РЕКОМЕНДУЕМАЯ СТАВКА: {self.suggested_trade_amount} руб.", color='magenta', style='bold')
        colored_print("", add_border=True)
        
        # Логируем в файл без вывода в консоль через логер
        file_handler = next((h for h in logger.handlers if isinstance(h, logging.FileHandler)), None)
        if file_handler:
            log_message = f"Сигнал: {action_text} ({confidence:.1f}%)\n{message}\nИсточники: {self.signal_sources}"
            file_handler.emit(logging.LogRecord(
                name=logger.name, 
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg=log_message,
                args=(),
                exc_info=None
            ))
        
    def _play_signal_sound(self, confidence, direction=None):
        """
        Воспроизведение звукового сигнала в зависимости от направления прогноза
        
        Args:
            confidence: уверенность в процентах
            direction: направление сигнала ('up' или 'down')
        """
        try:
            if direction == "up":
                # Сигнал на повышение
                play_wav("up.wav")
            elif direction == "down":
                # Сигнал на понижение
                play_wav("down.wav")
            else:
                # Если направление не указано, используем стандартный звук
                winsound.Beep(1000, 500)
        except Exception as e:
            logger.error(f"Ошибка при воспроизведении звука: {e}")
    
    def _log_binary_signal(self, action, confidence):
        """
        Логирование сигнала для бинарных опционов
        
        Args:
            action: действие (UP/DOWN)
            confidence: уверенность (0-100)
        """
        try:
            with open(self.binary_log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    self.current_pair,
                    self.timeframe,
                    self.expiration_time,
                    action,
                    f"{confidence:.1f}",
                    self.signal_sources,
                    "",  # Результат (будет заполнен позже)
                    ""   # P&L (будет заполнен позже)
                ])
        except Exception as e:
            logger.error(f"Ошибка при логировании сигнала: {e}")
    
    def update_result(self, signal_time, result, pnl):
        """
        Обновление результата сигнала
        
        Args:
            signal_time: время сигнала (строка в формате 'YYYY-MM-DD HH:MM:SS')
            result: результат (WIN/LOSS)
            pnl: прибыль/убыток
        """
        try:
            updated = False
            temp_file = f"{self.binary_log_file}.temp"
            
            with open(self.binary_log_file, 'r', newline='', encoding='utf-8') as input_file, \
                 open(temp_file, 'w', newline='', encoding='utf-8') as output_file:
                
                reader = csv.reader(input_file)
                writer = csv.writer(output_file)
                
                header = next(reader)
                writer.writerow(header)
                
                for row in reader:
                    if row[0] == signal_time:  # Найден сигнал с указанным временем
                        row[7] = result  # Обновляем результат
                        row[8] = str(pnl)  # Обновляем P&L
                        updated = True
                    writer.writerow(row)
            
            # Заменяем исходный файл на обновленный
            os.replace(temp_file, self.binary_log_file)
            
            if updated:
                logger.info(f"Результат сигнала от {signal_time} обновлен: {result}, P&L: {pnl}")
            else:
                logger.warning(f"Сигнал с временем {signal_time} не найден в логе")
                
        except Exception as e:
            logger.error(f"Ошибка при обновлении результата: {e}")
            
    def _schedule_prediction_check(self, signal_time, direction, expiration_minutes):
        """
        Планирует проверку прогноза через указанное время экспирации
        
        Args:
            signal_time: время сигнала
            direction: направление прогноза ('up' или 'down')
            expiration_minutes: время экспирации в минутах
        """
        check_time = expiration_minutes * 60  # секунды
        timer = threading.Timer(check_time, self._check_prediction_result, 
                               args=[signal_time, direction])
        timer.daemon = True
        timer.start()
        
        logger.info(f"Проверка результата запланирована через {expiration_minutes} мин.")
    
    def _check_prediction_result(self, signal_time, direction):
        """
        Проверяет результат прогноза
        
        Args:
            signal_time: время сигнала
            direction: направление прогноза ('up' или 'down')
        """
        if not self.running:
            return  # Если программа остановлена, не проверяем
            
        try:
            # Захват экрана для проверки текущей цены
            frame = capture_screen(self.monitor_area)
            current_candles = detect_candles(frame)
            
            if not current_candles:
                logger.warning("Не удалось обнаружить свечи для проверки прогноза")
                return
                
            # Получаем текущую цену (Y-координата центра последней свечи)
            current_price = current_candles[-1]["center_y"]
            
            # Определяем, был ли прогноз верным
            # Помним, что на экране Y растет вниз, поэтому логика инвертирована
            if direction == "up":
                # Прогноз "вверх" - цена должна быть ниже исходной
                result = current_price < self.prediction_starting_price
            else:
                # Прогноз "вниз" - цена должна быть выше исходной
                result = current_price > self.prediction_starting_price
                
            result_text = "WIN" if result else "LOSS"
            
            # Рассчитываем условный P&L
            pnl = self.suggested_trade_amount if result else -self.suggested_trade_amount
                
            # Обновляем статистику
            self.update_result(signal_time, result_text, pnl)
            
            # Обновляем лог паттернов
            pattern = self.signal_sources if self.signal_sources else "Комплексный сигнал"
            self.log_pattern_forecast(
                pattern=pattern, 
                confidence=self.signal_strength/100.0,
                direction=direction,
                is_correct=result,
                initial_price=self.prediction_starting_price,
                final_price=current_price
            )
                
            # Выводим результат на экран в более заметном формате
            self._print_prediction_result(signal_time, direction, result_text, pnl)
                
        except Exception as e:
            logger.error(f"Ошибка при проверке результата прогноза: {e}")
            
    def _print_prediction_result(self, signal_time, direction, result_text, pnl):
        """
        Отображает результат прогноза в цветном формате, без дублирования в логи
        """
        direction_text = "ВВЕРХ" if direction == "up" else "ВНИЗ"
        is_win = result_text == "WIN"
        result_display = "ВЫИГРЫШ" if is_win else "ПРОИГРЫШ"
        result_color = "green" if is_win else "red"
        self.play_message_sound()
        
        # Выводим результат с цветным форматированием
        colored_print("", add_border=True)
        colored_print("РЕЗУЛЬТАТ ПРОГНОЗА:", color='blue', style='bold')
        colored_print("")
        colored_print(f"Время сигнала: {signal_time}", color='cyan')
        colored_print(f"Направление: {direction_text}", color='cyan')
        colored_print(f"Результат: {result_display}", color=result_color, style='bold')
        colored_print(f"P&L: {pnl}", color='magenta')
        colored_print("", add_border=True)
        
        # Логируем в файл без вывода в консоль через логер
        file_handler = next((h for h in logger.handlers if isinstance(h, logging.FileHandler)), None)
        if file_handler:
            log_message = f"Результат прогноза: {direction_text} - {result_display}, P&L: {pnl}"
            file_handler.emit(logging.LogRecord(
                name=logger.name, 
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg=log_message,
                args=(),
                exc_info=None
            ))
    
    def log_pattern_forecast(self, pattern, confidence, direction, is_correct=None, initial_price=None, final_price=None):
        """
        Запись информации о прогнозе и паттерне в CSV-файл
        
        Args:
            pattern: название паттерна
            confidence: уверенность в прогнозе (0.0-1.0)
            direction: направление прогноза ("up" или "down")
            is_correct: верность прогноза (True/False/None)
            initial_price: начальная цена
            final_price: конечная цена
        """
        log_file = "pattern_forecast_log.csv"
        
        try:
            # Проверяем существование файла и создаем его с заголовками, если нужно
            if not os.path.exists(log_file):
                with open(log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'Время', 
                        'Паттерн', 
                        'Уверенность', 
                        'Направление', 
                        'Валютная пара',
                        'Таймфрейм',
                        'Экспирация',
                        'Начальная цена',
                        'Конечная цена',
                        'Результат', 
                        'Правильно',
                        'Источники сигнала'
                    ])
                logger.info(f"Создан новый файл логов по паттернам: {log_file}")
            
            # Форматирование результата
            result = ""
            if is_correct is not None:
                result = "Верно" if is_correct else "Неверно"
            
            # Запись в файл
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    pattern,
                    f"{confidence:.2f}",
                    "Вверх" if direction == "up" else "Вниз",
                    self.current_pair,
                    self.timeframe,
                    self.expiration_time,
                    initial_price or "",
                    final_price or "",
                    result,
                    str(is_correct) if is_correct is not None else "",
                    self.signal_sources
                ])
            
            logger.info(f"Запись о паттерне {pattern} добавлена в лог")
            return True
        except Exception as e:
            logger.error(f"Ошибка при записи в лог паттернов: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False    
        
    def start_binary_analysis(self):
        """Запуск анализа для бинарных опционов"""
        if not self.monitor_area:
            self.monitor_area = select_monitor_area()
            if not self.monitor_area:
                logger.warning("Область не выбрана. Анализ не запущен.")
                return
        
        self.running = True
        
        # Вывод без использования логгера для избежания префиксов
        print("\nSM запущен для бинарного анализа...")
        print(f"Валютная пара: {self.current_pair}")
        print(f"Таймфрейм: {self.timeframe} мин, Экспирация: {self.expiration_time} мин")
        print("Нажмите ESC для остановки, 'P' для приостановки уведомлений, 'C' для возобновления.")
        
        # Логируем в файл с полной информацией
        logger.info("SM запущен для бинарного анализа...")
        logger.info(f"Валютная пара: {self.current_pair}")
        logger.info(f"Таймфрейм: {self.timeframe} мин, Экспирация: {self.expiration_time} мин")
        
        # Закрываем все окна перед началом анализа
        cv2.destroyAllWindows()
        time.sleep(0.5)  # Даем время на закрытие окон
        
        # Создаем новое окно анализа
        cv2.namedWindow("SM - Binary Analysis", cv2.WINDOW_NORMAL)
        
        # Добавляем период разогрева 
        warming_period = 5  # Количество циклов анализа перед началом прогнозов
        current_cycle = 0
        print("Система анализирует график...")
        logger.info("Система анализирует график...")
        
        try:
            while self.running:
                # Захват экрана
                frame = capture_screen(self.monitor_area)
                
                # Обнаружение свечей
                detected_candles = detect_candles(frame)
                self.current_candles = detected_candles
                
                # Период разогрева - только анализируем, не делаем прогнозов
                if current_cycle < warming_period:
                    current_cycle += 1
                    display_binary_analysis(frame, detected_candles, None, 0, self.current_pair)
                    if current_cycle % 2 == 0:  # Уменьшаем количество сообщений
                        print(f"Сбор данных: {current_cycle}/{warming_period}")
                        logger.info(f"Сбор данных: {current_cycle}/{warming_period}")
                else:
                    # Комплексный анализ с использованием нескольких индикаторов
                    direction, confidence, is_enhanced = self.analyze_binary_signal(detected_candles)
                    
                    # Если есть сильный сигнал, оповещаем пользователя
                    if direction and confidence >= self.min_confidence_threshold and not self.paused:
                        action, message, amount = self.get_recommended_action()
                        if action != "HOLD":
                            self._notify_binary_signal(action, confidence * 100, message)
                    
                    # Отображение результатов анализа
                    display_binary_analysis(frame, detected_candles, direction, confidence * 100, self.current_pair)
                
                # Обработка нажатий клавиш
                key = cv2.waitKey(20) & 0xFF
                if key == 27:  # ESC для выхода
                    break
                elif key == ord('p') or key == ord('P'):  # 'P' для приостановки
                    self.pause_forecasts()
                elif key == ord('c') or key == ord('C'):  # 'C' для возобновления
                    self.resume_forecasts()
                elif key == ord('s') or key == ord('S'):  # 'S' для скриншота
                    take_screenshot(frame, "sm_binary")
                
                # Задержка для контроля частоты анализа
                time.sleep(0.2)
                    
        except KeyboardInterrupt:
           custom_print("\nАнализ остановлен пользователем.")
        except Exception as e:
            logger.error(f"Произошла ошибка при анализе: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            cv2.destroyAllWindows()
            self.running = False
            
    def pause_forecasts(self):
        """Приостановка выдачи прогнозов"""
        self.paused = True
        logger.info("Выдача прогнозов приостановлена")
        print("Выдача прогнозов приостановлена")

    def resume_forecasts(self):
        """Возобновление выдачи прогнозов"""
        self.paused = False
        logger.info("Выдача прогнозов возобновлена")
        print("Выдача прогнозов возобновлена")