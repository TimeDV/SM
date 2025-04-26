"""
Модуль для обнаружения японских свечей на графике в системе ScalpMaster (SM)
Отвечает за распознавание и анализ свечей
"""

import cv2
import numpy as np
import logging

# Настройка логирования
logger = logging.getLogger("candle_detection")

def detect_candle_color(roi):
    """
    Определение цвета свечи на основе изображения
    
    Args:
        roi: изображение области свечи (region of interest)
        
    Returns:
        str: "green", "red" или "neutral"
    """
    try:
        # Защита от пустых областей
        if roi is None or roi.size == 0:
            return "neutral"
        
        # Преобразуем в HSV для лучшего определения цвета
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Диапазон для зеленого цвета
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Диапазон для красного цвета (два диапазона из-за особенностей HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels = cv2.countNonZero(red_mask)
        
        # Определяем доминирующий цвет
        if green_pixels > red_pixels and green_pixels > 10:
            return "green"
        elif red_pixels > green_pixels and red_pixels > 10:
            return "red"
        else:
            return "neutral"
            
    except Exception as e:
        logger.error(f"Ошибка при определении цвета свечи: {e}")
        return "neutral"

def detect_candles(frame):
    """
    Обнаружение японских свечей на изображении с определением их цвета
    
    Args:
        frame: изображение, на котором ищем свечи
        
    Returns:
        list: список обнаруженных свечей с их параметрами
    """
    try:
        # Проверка размера изображения
        if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            logger.warning("Предупреждение: получено пустое изображение")
            return []
        
        # Создаем копию для анализа цвета
        color_frame = frame.copy()
        
        # Преобразование в оттенки серого для обнаружения контуров
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Применение фильтрации для выделения контуров
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Поиск контуров свечей
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candles = []
        for contour in contours:
            # Фильтрация контуров по размеру и форме
            if cv2.contourArea(contour) > 50:  # Минимальная площадь
                x, y, w, h = cv2.boundingRect(contour)
                if h > w * 1.5:  # Свечи обычно вытянуты вертикально
                    # Определяем верх и низ свечи
                    top_y = y
                    bottom_y = y + h
                    
                    # Анализируем цвет в области свечи
                    roi = color_frame[y:y+h, x:x+w]
                    candle_color = detect_candle_color(roi)
                    
                    # Добавляем свечу в список с информацией о цвете
                    candles.append({
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "top": top_y,
                        "bottom": bottom_y,
                        "center_x": x + w // 2,
                        "center_y": y + h // 2,
                        "contour": contour,
                        "color": candle_color
                    })
        
        # Сортировка свечей слева направо (по времени)
        candles.sort(key=lambda c: c["x"])
        
        # Добавляем дополнительные метрики для анализа
        if len(candles) >= 2:
            for i in range(1, len(candles)):
                # Определение направления и размера движения
                prev_candle = candles[i-1]
                curr_candle = candles[i]
                
                # Движение цены (отрицательное значение - рост, положительное - падение)
                # Помним, что в координатах экрана ось Y направлена вниз
                price_move = curr_candle["center_y"] - prev_candle["center_y"]
                curr_candle["price_move"] = price_move
                
                # Определение силы движения относительно высоты свечи
                curr_candle["move_strength"] = abs(price_move) / curr_candle["height"] if curr_candle["height"] > 0 else 0
        
        # Расчет дополнительных параметров для всех свечей
        for candle in candles:
            # Тело свечи (расстояние от верха до низа)
            candle["body_size"] = candle["height"]
            
            # Отношение тела к высоте (для определения пин-баров и других паттернов)
            if "move_strength" not in candle:
                candle["move_strength"] = 0
        
        logger.debug(f"Обнаружено свечей: {len(candles)}")
        return candles
        
    except Exception as e:
        logger.error(f"Ошибка при обнаружении свечей: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def preprocess_candles(candles, min_candles=5):
    """
    Предварительная обработка списка свечей для дальнейшего анализа
    
    Args:
        candles: список обнаруженных свечей
        min_candles: минимальное количество свечей для анализа
        
    Returns:
        list: обработанный список свечей или пустой список если данных недостаточно
        bool: флаг успешности обработки
    """
    try:
        if len(candles) < min_candles:
            logger.debug(f"Недостаточно свечей для анализа: {len(candles)} < {min_candles}")
            return candles, False
            
        # Удаление выбросов (свечи с аномальными размерами)
        heights = [c["height"] for c in candles]
        median_height = np.median(heights)
        stddev_height = np.std(heights)
        
        filtered_candles = []
        for candle in candles:
            # Фильтруем свечи, которые сильно отличаются по размеру
            if abs(candle["height"] - median_height) <= 2.5 * stddev_height:
                filtered_candles.append(candle)
        
        # Если после фильтрации осталось слишком мало свечей, используем исходный список
        if len(filtered_candles) < min_candles:
            logger.debug(f"После фильтрации осталось мало свечей: {len(filtered_candles)}, используем исходные")
            return candles, True
        
        # Нормализация параметров свечей
        min_y = min(c["top"] for c in filtered_candles)
        max_y = max(c["bottom"] for c in filtered_candles)
        height_range = max_y - min_y
        
        for candle in filtered_candles:
            # Нормализованная позиция верха и низа (0-1)
            if height_range > 0:
                candle["norm_top"] = (candle["top"] - min_y) / height_range
                candle["norm_bottom"] = (candle["bottom"] - min_y) / height_range
                candle["norm_center"] = (candle["center_y"] - min_y) / height_range
            else:
                candle["norm_top"] = candle["norm_bottom"] = candle["norm_center"] = 0.5
                
            # Определение типа свечи на основе цвета и размера
            if candle["color"] == "green":
                candle["type"] = "bullish"  # Бычья свеча
            elif candle["color"] == "red":
                candle["type"] = "bearish"  # Медвежья свеча
            else:
                candle["type"] = "neutral"  # Нейтральная свеча
        
        logger.debug(f"Предобработка завершена: {len(filtered_candles)} свечей")
        return filtered_candles, True
        
    except Exception as e:
        logger.error(f"Ошибка при предобработке свечей: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return candles, False

def find_local_extrema(candles, window_size=3):
    """
    Поиск локальных максимумов и минимумов в ряде свечей
    
    Args:
        candles: список свечей
        window_size: размер окна для поиска (нечетное число)
        
    Returns:
        tuple: (локальные минимумы, локальные максимумы)
        каждый элемент - список кортежей (индекс, значение, цвет)
    """
    if len(candles) < window_size:
        return [], []
    
    # Приводим размер окна к нечетному значению
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    local_mins = []
    local_maxs = []
    
    for i in range(half_window, len(candles) - half_window):
        values_before = [candles[i-j]["bottom"] for j in range(1, half_window+1)]
        values_after = [candles[i+j]["bottom"] for j in range(1, half_window+1)]
        
        # Поиск локальных минимумов (пики вниз)
        if all(candles[i]["bottom"] > v for v in values_before) and all(candles[i]["bottom"] > v for v in values_after):
            local_mins.append((i, candles[i]["bottom"], candles[i]["color"]))
        
        values_before = [candles[i-j]["top"] for j in range(1, half_window+1)]
        values_after = [candles[i+j]["top"] for j in range(1, half_window+1)]
        
        # Поиск локальных максимумов (пики вверх)
        if all(candles[i]["top"] < v for v in values_before) and all(candles[i]["top"] < v for v in values_after):
            local_maxs.append((i, candles[i]["top"], candles[i]["color"]))
    
    return local_mins, local_maxs

def analyze_trend(candles, window_size=5):
    """
    Анализ тренда на основе последовательности свечей
    
    Args:
        candles: список свечей
        window_size: размер окна для определения тренда
        
    Returns:
        str: направление тренда ("up", "down", "sideways")
        float: сила тренда (0.0-1.0)
    """
    if len(candles) < window_size * 2:
        return "sideways", 0.0
    
    # Берем последние N*2 свечей
    recent_candles = candles[-window_size*2:]
    
    # Рассчитываем линейную регрессию для центров свечей
    x = np.array([i for i in range(len(recent_candles))])
    y = np.array([c["center_y"] for c in recent_candles])
    
    slope, intercept = np.polyfit(x, y, 1)
    
    # Вычисляем r_squared (коэффициент детерминации)
    y_pred = slope * x + intercept
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Подсчет свечей разных цветов
    green_count = sum(1 for c in recent_candles if c["color"] == "green")
    red_count = sum(1 for c in recent_candles if c["color"] == "red")
    
    # Определение направления и силы тренда
    trend_strength = min(1.0, abs(slope) * 10)  # Нормализуем силу тренда
    
    # На экране ось Y направлена вниз, поэтому slope < 0 означает восходящий тренд
    if slope < 0 and r_squared > 0.6:
        # Проверяем подтверждение цветом свечей
        if green_count > red_count:
            trend_strength *= (1 + green_count / len(recent_candles)) / 2
        else:
            trend_strength *= 0.7  # Снижаем уверенность при противоречии цветов
        return "up", trend_strength
    elif slope > 0 and r_squared > 0.6:
        # Проверяем подтверждение цветом свечей
        if red_count > green_count:
            trend_strength *= (1 + red_count / len(recent_candles)) / 2
        else:
            trend_strength *= 0.7  # Снижаем уверенность при противоречии цветов
        return "down", trend_strength
    else:
        # Боковой тренд или нет четкого тренда
        return "sideways", r_squared * 0.5  # Сила бокового тренда - половина r_squared