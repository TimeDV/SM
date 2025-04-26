"""
Модуль визуализации для ScalpMaster (SM)
Содержит функции для отображения анализа на экране
"""

import cv2
import numpy as np
import datetime
import logging
from pattern_detection import draw_trendline

# Настройка логирования
logger = logging.getLogger("visualization")

def display_analysis(frame, candles, pattern=None, confidence=0.0, analyzer=None):
    """
    Отображение анализа на экране
    
    Args:
        frame: исходное изображение
        candles: список обнаруженных свечей
        pattern: название паттерна (если обнаружен)
        confidence: уверенность в паттерне (0.0-1.0)
        analyzer: экземпляр анализатора (для доступа к дополнительной информации)
    """
    try:
        # Копия изображения для рисования
        display = frame.copy()
        
        # Рисуем обнаруженные свечи
        for candle in candles:
            # Определяем цвет контура в зависимости от цвета свечи
            if candle.get("color") == "green":
                contour_color = (0, 255, 0)  # Зеленый для восходящих свечей
            elif candle.get("color") == "red":
                contour_color = (0, 0, 255)  # Красный для нисходящих свечей
            else:
                contour_color = (255, 255, 255)  # Белый для нейтральных
                
            cv2.drawContours(display, [candle.get("contour", np.array([]))], -1, contour_color, 2)
        
        # Если есть информация о линии тренда, отрисовываем её
        trendline_info = getattr(analyzer, 'trendline_info', None) if pattern == "Линия тренда" else None
        if trendline_info:
            draw_trendline(display, candles, trendline_info)
        
        # Создаем информационную панель
        info_panel_height = 100  # Высота информационной панели
        info_panel = np.zeros((info_panel_height, display.shape[1], 3), dtype=np.uint8)
        
        # Заполняем информационную панель градиентом
        for i in range(info_panel_height):
            alpha = i / info_panel_height
            info_panel[i, :] = (50 + int(30 * alpha), 50 + int(30 * alpha), 50 + int(30 * alpha))
        
        # Добавляем информацию о времени
        current_time_str = datetime.datetime.now().strftime('%H:%M:%S')
        cv2.putText(info_panel, f"Time: {current_time_str}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Добавляем информацию о таймфрейме (получим из текущего режима анализа)
        try:
            candle_time = getattr(analyzer, 'candle_time', 60) // 60  # Конвертируем секунды в минуты
            forecast_time = getattr(analyzer, 'forecast_time', 3)
            cv2.putText(info_panel, f"Timeframe: {candle_time} min / Forecast: {forecast_time} min", (10, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except:
            # Если не удалось получить настройки, используем значения по умолчанию
            cv2.putText(info_panel, "Timeframe: 1 min / Forecast: 3 min", (10, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Добавляем информацию о паттерне если он обнаружен
        if pattern:
            # Перевод русских названий в английские
            pattern_eng = {
                "Двойное дно": "Double Bottom",
                "Двойная вершина": "Double Top",
                "Восходящий клин": "Rising Wedge",
                "Нисходящий клин": "Falling Wedge",
                "Линия тренда": "Trendline"
            }.get(pattern, pattern)
            
            pattern_color = (0, 255, 255)  # Желтый цвет для паттерна
            cv2.putText(info_panel, f"Pattern: {pattern_eng} ({confidence:.2f})", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, pattern_color, 2)
        
        # Объединяем основное изображение с информационной панелью
        combined_display = np.vstack((display, info_panel))
        
        # Отображаем результат
        cv2.imshow("Forex Pattern Analysis", combined_display)
        
    except Exception as e:
        logger.error(f"Ошибка при отображении анализа: {e}")
        import traceback
        logger.error(traceback.format_exc())

def display_binary_analysis(frame, candles, signal_direction, signal_strength, currency_pair):
    """
    Отображение анализа для бинарных опционов
    
    Args:
        frame: исходное изображение
        candles: список обнаруженных свечей
        signal_direction: направление сигнала ("up", "down" или None)
        signal_strength: сила сигнала (0-100)
        currency_pair: текущая валютная пара
    """
    try:
        # Копия изображения для рисования
        display = frame.copy()
        
        # Рисуем обнаруженные свечи
        for candle in candles:
            # Определяем цвет контура в зависимости от цвета свечи
            if candle.get("color") == "green":
                contour_color = (0, 255, 0)  # Зеленый для восходящих свечей
            elif candle.get("color") == "red":
                contour_color = (0, 0, 255)  # Красный для нисходящих свечей
            else:
                contour_color = (255, 255, 255)  # Белый для нейтральных
                
            cv2.drawContours(display, [candle.get("contour", np.array([]))], -1, contour_color, 2)
        
        # Высота информационной панели
        info_panel_height = 150
        
        # Создаем информационную панель с градиентным фоном
        info_panel = np.zeros((info_panel_height, display.shape[1], 3), dtype=np.uint8)
        
        # Заполняем информационную панель градиентом
        for i in range(info_panel_height):
            alpha = i / info_panel_height
            info_panel[i, :] = (50 + int(30 * alpha), 50 + int(30 * alpha), 50 + int(30 * alpha))
        
        # Добавляем информацию о времени
        current_time_str = datetime.datetime.now().strftime('%H:%M:%S')
        put_text_with_encoding(info_panel, f"Time: {current_time_str}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Добавляем информацию о валютной паре
        pair_text = f"Currency pair: {currency_pair}" if currency_pair else "Currency pair: not detected"
        put_text_with_encoding(info_panel, pair_text, (display.shape[1] - 300, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Добавляем разделительную линию
        cv2.line(info_panel, (0, 40), (display.shape[1], 40), (200, 200, 200), 1)
        
        # Добавляем информацию о сигнале
        if signal_direction:
            if signal_direction == "up":
                signal_color = (0, 255, 0)  # Зеленый для сигнала вверх
                signal_text = "Up"
            else:
                signal_color = (0, 0, 255)  # Красный для сигнала вниз
                signal_text = "Down"
            
            # Отображаем направление сигнала
            put_text_with_encoding(info_panel, f"Signal: {signal_text}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, signal_color, 2)
            
            # Отображаем силу сигнала в процентах
            put_text_with_encoding(info_panel, f"Confidence: {signal_strength:.1f}%", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Добавляем визуальную шкалу силы сигнала
            bar_length = int((display.shape[1] - 300) * (signal_strength / 100))
            cv2.rectangle(info_panel, (280, 110), (280 + bar_length, 130), signal_color, -1)
            cv2.rectangle(info_panel, (280, 110), (display.shape[1] - 20, 130), (150, 150, 150), 1)
        else:
            # Если сигнала нет
            put_text_with_encoding(info_panel, "No clear signal", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2)
        
        # Объединяем основное изображение с информационной панелью
        combined_display = np.vstack((display, info_panel))
        
        # Отображаем результат
        cv2.imshow("SM - Binary Analysis", combined_display)
                
    except Exception as e:
        logger.error(f"Ошибка при отображении анализа: {e}")
        import traceback
        logger.error(traceback.format_exc())

def put_text_with_encoding(img, text, position, font, scale, color, thickness):
    """
    Отображение текста с поддержкой кириллицы и других unicode-символов
    """
    try:
        # Создаем временное изображение для текста (черное)
        text_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        
        # Наносим текст на временное изображение (белым цветом)
        cv2.putText(text_img, text, position, font, scale, (255, 255, 255), thickness)
        
        # Конвертируем изображение в черно-белое
        text_gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
        
        # Создаем маску из текста
        _, mask = cv2.threshold(text_gray, 10, 255, cv2.THRESH_BINARY)
        
        # Накладываем текст с нужным цветом на исходное изображение
        roi = img[0:img.shape[0], 0:img.shape[1]]
        roi[mask > 0] = color
        
        return img
    except Exception as e:
        logger.error(f"Ошибка при отображении текста: {e}")
        return img
