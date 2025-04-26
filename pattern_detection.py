"""
Единый модуль обнаружения паттернов для системы ScalpMaster (SM)
Объединяет все алгоритмы обнаружения паттернов на графике
"""

import numpy as np
import cv2
import logging
import warnings

# Исправление для совместимости с новыми версиями NumPy
try:
    from numpy.polynomial.polynomial import RankWarning
    warnings.filterwarnings("ignore", category=RankWarning)
except (ImportError, AttributeError):
    # Для совместимости со старыми версиями NumPy
    try:
        warnings.filterwarnings("ignore", category=np.RankWarning)
    except AttributeError:
        # Если RankWarning недоступен вообще
        warnings.filterwarnings("ignore", category=UserWarning)

from candle_detection import find_local_extrema, analyze_trend, preprocess_candles

# Настройка логирования
logger = logging.getLogger("pattern_detection")

def perform_linear_regression(points):
    """
    Выполнение линейной регрессии для набора точек
    
    Args:
        points: список точек в формате [(x1, y1), (x2, y2), ...]
        
    Returns:
        tuple: (наклон, пересечение, r_squared)
    """
    try:
        if len(points) < 2:
            return 0, 0, 0
            
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        
        # Линейная регрессия
        slope, intercept = np.polyfit(x, y, 1)
        
        # Вычисляем r_squared (коэффициент детерминации)
        y_pred = slope * x + intercept
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return slope, intercept, r_squared
    except Exception as e:
        logger.error(f"Ошибка при выполнении линейной регрессии: {e}")
        return 0, 0, 0

def check_trend_touches(candles, slope, intercept, touch_threshold=0.2):
    """
    Проверка количества касаний линии тренда
    
    Args:
        candles: список свечей
        slope: наклон линии тренда
        intercept: пересечение линии тренда
        touch_threshold: порог для определения касания
        
    Returns:
        int: количество касаний
        list: индексы свечей, касающихся линии тренда
    """
    touches = 0
    touch_indices = []
    
    for i, candle in enumerate(candles):
        # Расчет ожидаемого значения Y на линии тренда для данного X
        expected_y = float(slope) * candle["center_x"] + float(intercept)
        
        # Расстояние от центра свечи до линии тренда
        distance = abs(candle["center_y"] - expected_y)
        max_distance = candle["height"] * touch_threshold
        
        # Если свеча находится близко к линии тренда, считаем это касанием
        if distance < max_distance:
            touches += 1
            touch_indices.append(i)
    
    return touches, touch_indices

def detect_double_bottom(candles, threshold=0.2):
    """
    Обнаружение паттерна "Двойное дно" с улучшенным алгоритмом
    
    Args:
        candles: список обнаруженных свечей
        threshold: порог для определения сходства минимумов
        
    Returns:
        bool: обнаружен ли паттерн
        float: уверенность в паттерне (0.0-1.0)
    """
    try:
        # Предварительная обработка свечей
        processed_candles, success = preprocess_candles(candles)
        if not success or len(processed_candles) < 7:
            return False, 0.0
        
        # Поиск локальных минимумов
        local_mins, _ = find_local_extrema(processed_candles)
        
        # Нужно минимум 2 локальных минимума
        if len(local_mins) < 2:
            return False, 0.0
        
        # Улучшенный алгоритм для двойного дна
        best_confidence = 0.0
        for i in range(len(local_mins) - 1):
            for j in range(i + 1, len(local_mins)):
                # Проверка, что минимумы на одном уровне
                min1_idx, min1_y, min1_color = local_mins[i]
                min2_idx, min2_y, min2_color = local_mins[j]
                
                # Защита от деления на ноль
                max_val = max(min1_y, min2_y)
                if max_val == 0:
                    continue
                
                # Расстояние между минимумами (в индексах свечей)
                distance = min2_idx - min1_idx
                
                # Проверка условий паттерна
                level_similarity = 1.0 - abs(min1_y - min2_y) / max_val
                if level_similarity > (1.0 - threshold) and distance >= 3 and distance <= len(processed_candles) // 2:
                    # Проверка наличия вершины между минимумами
                    has_peak = False
                    peak_height = 0
                    peak_index = 0
                    
                    for k in range(min1_idx + 1, min2_idx):
                        # Вершина должна быть выше обоих минимумов
                        if processed_candles[k]["top"] < min(min1_y, min2_y):
                            has_peak = True
                            # Находим высоту вершины относительно минимумов
                            current_height = min(min1_y, min2_y) - processed_candles[k]["top"]
                            if current_height > peak_height:
                                peak_height = current_height
                                peak_index = k
                    
                    if has_peak:
                        # Проверка цветов свечей для типичного двойного дна
                        color_score = 0.0
                        
                        # Проверка цвета первого минимума (предпочтительно красный)
                        if min1_color == "red":
                            color_score += 0.3
                        
                        # Проверка цвета центральной вершины (предпочтительно зеленый)
                        if peak_index > 0 and processed_candles[peak_index]["color"] == "green":
                            color_score += 0.4
                        
                        # Проверка последних свечей после второго минимума (предпочтительно зеленые)
                        if min2_idx + 1 < len(processed_candles) and processed_candles[min2_idx + 1]["color"] == "green":
                            color_score += 0.3
                        
                        # Высота пика влияет на уверенность
                        peak_factor = min(1.0, peak_height / (max_val * 0.2))
                        
                        # Расчет уверенности на основе сходства минимумов, высоты пика и цветов свечей
                        confidence = 0.3 * level_similarity + 0.4 * peak_factor + 0.3 * color_score
                        
                        # Бонус за более чёткие W-образные формы
                        if level_similarity > 0.9 and peak_factor > 0.7:
                            confidence = min(0.95, confidence * 1.1)
                        
                        # Дополнительная проверка на устойчивость паттерна
                        if min2_idx >= len(processed_candles) - 3:
                            # Паттерн формируется прямо сейчас, повышаем уверенность
                            confidence = min(0.95, confidence * 1.05)
                        
                        best_confidence = max(best_confidence, confidence)
        
        return best_confidence > threshold, best_confidence
        
    except Exception as e:
        logger.error(f"Ошибка при обнаружении двойного дна: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, 0.0

def detect_double_top(candles, threshold=0.2):
    """
    Обнаружение паттерна "Двойная вершина" с улучшенным алгоритмом
    
    Args:
        candles: список обнаруженных свечей
        threshold: порог для определения сходства максимумов
        
    Returns:
        bool: обнаружен ли паттерн
        float: уверенность в паттерне (0.0-1.0)
    """
    try:
        # Предварительная обработка свечей
        processed_candles, success = preprocess_candles(candles)
        if not success or len(processed_candles) < 7:
            return False, 0.0
        
        # Поиск локальных максимумов
        _, local_maxs = find_local_extrema(processed_candles)
        
        # Нужно минимум 2 локальных максимума
        if len(local_maxs) < 2:
            return False, 0.0
        
        # Улучшенный алгоритм для двойной вершины
        best_confidence = 0.0
        for i in range(len(local_maxs) - 1):
            for j in range(i + 1, len(local_maxs)):
                # Проверка, что максимумы на одном уровне
                max1_idx, max1_y, max1_color = local_maxs[i]
                max2_idx, max2_y, max2_color = local_maxs[j]
                
                # Защита от деления на ноль
                min_val = min(max1_y, max2_y)
                if min_val == 0:
                    continue
                
                # Расстояние между максимумами
                distance = max2_idx - max1_idx
                
                # Проверка условий паттерна
                level_similarity = 1.0 - abs(max1_y - max2_y) / min_val
                if level_similarity > (1.0 - threshold) and distance >= 3 and distance <= len(processed_candles) // 2:
                    # Проверка наличия впадины между максимумами
                    has_valley = False
                    valley_depth = 0
                    valley_index = 0
                    
                    for k in range(max1_idx + 1, max2_idx):
                        # Впадина должна быть ниже обоих максимумов
                        if processed_candles[k]["bottom"] > max(max1_y, max2_y):
                            has_valley = True
                            # Находим глубину впадины относительно максимумов
                            current_depth = processed_candles[k]["bottom"] - max(max1_y, max2_y)
                            if current_depth > valley_depth:
                                valley_depth = current_depth
                                valley_index = k
                    
                    if has_valley:
                        # Проверка цветов свечей для типичной двойной вершины
                        color_score = 0.0
                        
                        # Проверка цвета первого максимума (предпочтительно зеленый)
                        if max1_color == "green":
                            color_score += 0.3
                        
                        # Проверка цвета центральной впадины (предпочтительно красный)
                        if valley_index > 0 and processed_candles[valley_index]["color"] == "red":
                            color_score += 0.4
                        
                        # Проверка последних свечей после второго максимума (предпочтительно красные)
                        if max2_idx + 1 < len(processed_candles) and processed_candles[max2_idx + 1]["color"] == "red":
                            color_score += 0.3
                        
                        # Глубина впадины влияет на уверенность
                        valley_factor = min(1.0, valley_depth / (min_val * 0.2))
                        
                        # Расчет уверенности
                        confidence = 0.3 * level_similarity + 0.4 * valley_factor + 0.3 * color_score
                        
                        # Бонус за более чёткие M-образные формы
                        if level_similarity > 0.9 and valley_factor > 0.7:
                            confidence = min(0.95, confidence * 1.1)
                            
                        # Дополнительная проверка на устойчивость паттерна
                        if max2_idx >= len(processed_candles) - 3:
                            # Паттерн формируется прямо сейчас, повышаем уверенность
                            confidence = min(0.95, confidence * 1.05)
                        
                        best_confidence = max(best_confidence, confidence)
        
        return best_confidence > threshold, best_confidence
        
    except Exception as e:
        logger.error(f"Ошибка при обнаружении двойной вершины: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, 0.0

def detect_ascending_wedge(candles, min_points=4, max_angle=30, threshold=0.25):
    """
    Обнаружение паттерна "Восходящий клин" с улучшенным алгоритмом
    
    Args:
        candles: список обнаруженных свечей
        min_points: минимальное количество точек для определения линий тренда
        max_angle: максимальный угол между линиями тренда (в градусах)
        threshold: порог уверенности для определения паттерна
        
    Returns:
        bool: обнаружен ли паттерн
        float: уверенность в паттерне (0.0-1.0)
    """
    try:
        # Предварительная обработка свечей
        processed_candles, success = preprocess_candles(candles)
        if not success or len(processed_candles) < min_points * 2:
            return False, 0.0
        
        # Находим верхние и нижние точки для построения линий тренда
        _, local_maxs = find_local_extrema(processed_candles)
        local_mins, _ = find_local_extrema(processed_candles)
        
        # Проверяем, достаточно ли точек
        if len(local_maxs) < min_points or len(local_mins) < min_points:
            return False, 0.0
            
        # Проверяем, что у нас есть как минимум 2 точки для каждой линии
        if len(local_maxs) < 2 or len(local_mins) < 2:
            return False, 0.0
        
        # Формируем точки для линий тренда
        upper_points = [(processed_candles[idx]["center_x"], y) for idx, y, _ in local_maxs]
        lower_points = [(processed_candles[idx]["center_x"], y) for idx, y, _ in local_mins]
        
        # Проверка восходящего тренда (на экране Y растет вниз, поэтому условие инвертировано)
        if not (upper_points[-1][1] < upper_points[0][1] and lower_points[-1][1] < lower_points[0][1]):
            return False, 0.0
        
        # Рассчитываем линии тренда
        upper_slope, upper_intercept, upper_r2 = perform_linear_regression(upper_points)
        lower_slope, lower_intercept, lower_r2 = perform_linear_regression(lower_points)
        
        # В восходящем клине обе линии должны иметь отрицательный наклон, но нижняя линия должна быть более крутой
        if not (upper_slope < 0 and lower_slope < 0 and abs(lower_slope) > abs(upper_slope)):
            return False, 0.0
        
        # Расчет угла между линиями
        angle = abs(np.degrees(np.arctan(upper_slope) - np.arctan(lower_slope)))
        
        if angle > max_angle:
            return False, 0.0
        
        # Проверка точек касания
        upper_touches, _ = check_trend_touches(processed_candles, upper_slope, upper_intercept)
        lower_touches, _ = check_trend_touches(processed_candles, lower_slope, lower_intercept)
        
        # Требуется минимум 2 касания для каждой границы
        if upper_touches < 2 or lower_touches < 2:
            return False, 0.0
        
        # Подсчет цветов свечей внутри клина
        green_candles = 0
        red_candles = 0
        total_candles = 0
        
        for candle in processed_candles:
            x = candle["center_x"]
            y = candle["center_y"]
            
            # Расчет ожидаемых значений верхней и нижней границы в точке свечи
            upper_expected = upper_slope * x + upper_intercept
            lower_expected = lower_slope * x + lower_intercept
            
            # Проверка, находится ли свеча между линиями тренда
            if lower_expected <= y <= upper_expected:
                total_candles += 1
                if candle["color"] == "green":
                    green_candles += 1
                elif candle["color"] == "red":
                    red_candles += 1
        
        # Для восходящего клина характерно преобладание зеленых свечей
        color_score = 0.0
        if total_candles > 0:
            # В восходящем клине должны преобладать зеленые свечи
            green_ratio = green_candles / total_candles
            
            # Идеальный восходящий клин имеет 60-80% зеленых свечей
            if 0.6 <= green_ratio <= 0.8:
                color_score = 1.0
            elif green_ratio > 0.8:
                color_score = 0.8  # Слишком много зеленых - менее характерно
            else:
                color_score = green_ratio  # Меньше зеленых - меньше уверенность
        
        # Расчет уверенности
        r2_quality = (upper_r2 + lower_r2) / 2  # Качество аппроксимации линий
        angle_quality = 1.0 - angle / max_angle  # Качество угла
        touch_quality = min(1.0, (upper_touches + lower_touches) / (min_points * 2))  # Качество касаний
        
        confidence = 0.25 * r2_quality + 0.25 * angle_quality + 0.25 * touch_quality + 0.25 * color_score
        
        return confidence > threshold, min(0.95, confidence)
        
    except Exception as e:
        logger.error(f"Ошибка при обнаружении восходящего клина: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, 0.0

def detect_descending_wedge(candles, min_points=4, max_angle=30, threshold=0.25):
    """
    Обнаружение паттерна "Нисходящий клин" с улучшенным алгоритмом
    
    Args:
        candles: список обнаруженных свечей
        min_points: минимальное количество точек для определения линий тренда
        max_angle: максимальный угол между линиями тренда (в градусах)
        threshold: порог уверенности для определения паттерна
        
    Returns:
        bool: обнаружен ли паттерн
        float: уверенность в паттерне (0.0-1.0)
    """
    try:
        # Предварительная обработка свечей
        processed_candles, success = preprocess_candles(candles)
        if not success or len(processed_candles) < min_points * 2:
            return False, 0.0
        
        # Находим верхние и нижние точки для построения линий тренда
        _, local_maxs = find_local_extrema(processed_candles)
        local_mins, _ = find_local_extrema(processed_candles)
        
        # Проверяем, достаточно ли точек
        if len(local_maxs) < min_points or len(local_mins) < min_points:
            return False, 0.0
            
        # Проверяем, что у нас есть как минимум 2 точки для каждой линии
        if len(local_maxs) < 2 or len(local_mins) < 2:
            return False, 0.0
        
        # Формируем точки для линий тренда
        upper_points = [(processed_candles[idx]["center_x"], y) for idx, y, _ in local_maxs]
        lower_points = [(processed_candles[idx]["center_x"], y) for idx, y, _ in local_mins]
        
        # Проверка нисходящего тренда (на экране Y растет вниз, поэтому условие инвертировано)
        if not (upper_points[-1][1] > upper_points[0][1] and lower_points[-1][1] > lower_points[0][1]):
            return False, 0.0
        
        # Рассчитываем линии тренда
        upper_slope, upper_intercept, upper_r2 = perform_linear_regression(upper_points)
        lower_slope, lower_intercept, lower_r2 = perform_linear_regression(lower_points)
        
        # В нисходящем клине обе линии должны иметь положительный наклон, но верхняя линия должна быть более крутой
        if not (upper_slope > 0 and lower_slope > 0 and upper_slope > lower_slope):
            return False, 0.0
        
        # Расчет угла между линиями
        angle = abs(np.degrees(np.arctan(upper_slope) - np.arctan(lower_slope)))
        
        if angle > max_angle:
            return False, 0.0
        
        # Проверка точек касания
        upper_touches, _ = check_trend_touches(processed_candles, upper_slope, upper_intercept)
        lower_touches, _ = check_trend_touches(processed_candles, lower_slope, lower_intercept)
        
        # Требуется минимум 2 касания для каждой границы
        if upper_touches < 2 or lower_touches < 2:
            return False, 0.0
        
        # Подсчет цветов свечей внутри клина
        green_candles = 0
        red_candles = 0
        total_candles = 0
        
        for candle in processed_candles:
            x = candle["center_x"]
            y = candle["center_y"]
            
            # Расчет ожидаемых значений верхней и нижней границы в точке свечи
            upper_expected = upper_slope * x + upper_intercept
            lower_expected = lower_slope * x + lower_intercept
            
            # Проверка, находится ли свеча между линиями тренда
            if lower_expected <= y <= upper_expected:
                total_candles += 1
                if candle["color"] == "green":
                    green_candles += 1
                elif candle["color"] == "red":
                    red_candles += 1
        
        # Для нисходящего клина характерно преобладание красных свечей
        color_score = 0.0
        if total_candles > 0:
            # В нисходящем клине должны преобладать красные свечи
            red_ratio = red_candles / total_candles
            
            # Идеальный нисходящий клин имеет 60-80% красных свечей
            if 0.6 <= red_ratio <= 0.8:
                color_score = 1.0
            elif red_ratio > 0.8:
                color_score = 0.8  # Слишком много красных - менее характерно
            else:
                color_score = red_ratio  # Меньше красных - меньше уверенность
        
        # Расчет уверенности
        r2_quality = (upper_r2 + lower_r2) / 2  # Качество аппроксимации линий
        angle_quality = 1.0 - angle / max_angle  # Качество угла
        touch_quality = min(1.0, (upper_touches + lower_touches) / (min_points * 2))  # Качество касаний
        
        confidence = 0.25 * r2_quality + 0.25 * angle_quality + 0.25 * touch_quality + 0.25 * color_score
        
        return confidence > threshold, min(0.95, confidence)
        
    except Exception as e:
        logger.error(f"Ошибка при обнаружении нисходящего клина: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, 0.0

def detect_trendline(candles, min_points=3, min_quality=0.5, min_angle=5, max_angle=45, threshold=0.15):
    """
    Обнаружение паттерна "Линия тренда" с улучшенным алгоритмом
    
    Args:
        candles: список обнаруженных свечей
        min_points: минимальное количество точек для определения тренда
        min_quality: минимальный коэффициент качества (r_squared)
        min_angle: минимальный угол наклона (градусы)
        max_angle: максимальный угол наклона (градусы)
        threshold: порог уверенности для определения паттерна
        
    Returns:
        bool: обнаружен ли паттерн
        float: уверенность в паттерне (0.0-1.0)
        dict: информация о линии тренда
    """
    try:
        # Предварительная обработка свечей
        processed_candles, success = preprocess_candles(candles)
        if not success or len(processed_candles) < min_points:
            return False, 0.0, {}
        
        # Используем последние N свечей для определения тренда
        recent_candles = processed_candles[-min_points:]
        
        # Извлекаем координаты центров свечей
        points = [(c["center_x"], c["center_y"]) for c in recent_candles]
        
        # Линейная регрессия для определения тренда
        slope, intercept, r_squared = perform_linear_regression(points)
        
        # Вычисляем угол наклона линии тренда (в градусах)
        angle_degrees = abs(np.degrees(np.arctan(slope)))
        
        # Проверяем критерии для определения линии тренда
        if r_squared < min_quality or angle_degrees < min_angle or angle_degrees > max_angle:
            return False, 0.0, {}
        
        # Определяем направление тренда
        # Помним, что на экране ось Y направлена вниз
        direction = "up" if slope < 0 else "down"
        
        # Проверка точек касания
        touches, touch_indices = check_trend_touches(processed_candles, slope, intercept)
        
        # Требуется минимум 3 касания для линии тренда
        if touches < 3:
            return False, 0.0, {}
        
        # Проверяем цвета свечей для подтверждения тренда
        green_count = sum(1 for c in recent_candles if c["color"] == "green")
        red_count = sum(1 for c in recent_candles if c["color"] == "red")
        
        # Для восходящего тренда должно быть больше зеленых свечей
        # Для нисходящего тренда должно быть больше красных свечей
        color_match = False
        if direction == "up" and green_count > red_count:
            color_match = True
        elif direction == "down" and red_count > green_count:
            color_match = True
        
        # Вычисляем уверенность в тренде
        confidence_quality = min(1.0, float(r_squared) / min_quality)
        
        # Вклад от угла (предпочитаем средние углы наклона)
        angle_factor = 0.0
        if min_angle <= angle_degrees <= max_angle:
            if angle_degrees <= 15:
                angle_factor = angle_degrees / 15.0  # От 0 до 1.0 при углах от min_angle до 15°
            elif angle_degrees <= 30:
                angle_factor = 1.0  # Максимум при углах 15-30°
            else:
                angle_factor = 1.0 - (angle_degrees - 30) / (max_angle - 30)  # Убывает от 1.0 до 0 при углах от 30° до max_angle
        
        # Вклад от длительности тренда
        duration_factor = min(1.0, len(recent_candles) / (min_points * 2))
        
        # Общая уверенность как взвешенная сумма факторов
        confidence = 0.5 * confidence_quality + 0.3 * angle_factor + 0.2 * duration_factor
        
        # Если цвета свечей не соответствуют направлению тренда, снижаем уверенность
        if not color_match:
            confidence *= 0.8
        
        # Дополнительная информация о тренде
        trendline_info = {
            "direction": direction,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "angle_degrees": float(angle_degrees),
            "points": len(recent_candles),
            "touches": touches,
            "touch_indices": touch_indices,
            "is_color_match": color_match
        }
        
        # Применяем порог чувствительности
        if confidence < threshold:
            return False, 0.0, {}
        
        return True, min(0.95, confidence), trendline_info
    
    except Exception as e:
        logger.error(f"Ошибка при обнаружении линии тренда: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, 0.0, {}

def draw_trendline(display, candles, trendline_info, color=(0, 255, 255), thickness=2):
    """
    Отрисовка линии тренда на изображении
    
    Args:
        display: изображение для отрисовки
        candles: список свечей
        trendline_info: информация о линии тренда
        color: цвет линии
        thickness: толщина линии
    """
    try:
        if not trendline_info or "slope" not in trendline_info:
            return
        
        # Получаем параметры линии тренда
        slope = trendline_info["slope"]
        intercept = trendline_info["intercept"]
        
        # Находим границы для отрисовки линии
        if candles:
            # Используем X-координаты первой и последней свечи
            x1 = candles[0]["center_x"]
            x2 = candles[-1]["center_x"]
            
            # Продлеваем линию немного вперед
            x2 += (x2 - x1) * 0.2
            
            # Вычисляем соответствующие Y-координаты
            y1 = int(slope * x1 + intercept)
            y2 = int(slope * x2 + intercept)
            
            # Рисуем линию
            cv2.line(display, (int(x1), y1), (int(x2), y2), color, thickness)
            
            # Добавляем небольшую стрелку в конце линии для указания направления
            arrow_length = 15
            arrow_angle = np.arctan(slope)
            
            # Координаты конца стрелки
            end_x = int(x2)
            end_y = int(y2)
            
            # Координаты крыльев стрелки
            arrow_x1 = int(end_x - arrow_length * np.cos(arrow_angle + np.pi/6))
            arrow_y1 = int(end_y - arrow_length * np.sin(arrow_angle + np.pi/6))
            arrow_x2 = int(end_x - arrow_length * np.cos(arrow_angle - np.pi/6))
            arrow_y2 = int(end_y - arrow_length * np.sin(arrow_angle - np.pi/6))
            
            # Рисуем стрелку
            cv2.line(display, (end_x, end_y), (arrow_x1, arrow_y1), color, thickness)
            cv2.line(display, (end_x, end_y), (arrow_x2, arrow_y2), color, thickness)
            
    except Exception as e:
        logger.error(f"Ошибка при отрисовке линии тренда: {e}")

def detect_trendline_signal(candles, min_points=6):
    """
    Обнаружение сигнала на основе линии тренда для бинарных опционов
    
    Args:
        candles: список свечей
        min_points: минимальное количество точек для определения тренда
        
    Returns:
        tuple: (направление, уверенность)
    """
    is_trendline, confidence, trendline_info = detect_trendline(candles, min_points=min_points)
    
    if not is_trendline:
        return None, 0
    
    # Для бинарных опционов важно убедиться в продолжении тренда
    direction = trendline_info.get("direction", None)
    if not direction:
        return None, 0
    
    # Проверка последних свечей на соответствие тренду
    recent_candles = candles[-3:]
    green_count = sum(1 for c in recent_candles if c["color"] == "green")
    red_count = sum(1 for c in recent_candles if c["color"] == "red")
    
    # Для восходящего тренда должны преобладать зеленые свечи
    # Для нисходящего тренда должны преобладать красные свечи
    recent_trend_strength = 0
    if direction == "up" and green_count > red_count:
        recent_trend_strength = green_count / len(recent_candles)
    elif direction == "down" and red_count > green_count:
        recent_trend_strength = red_count / len(recent_candles)
    else:
        # Последние свечи противоречат тренду - снижаем уверенность
        confidence *= 0.5
    
    # Корректируем уверенность с учетом силы последних свечей
    final_confidence = confidence * (0.7 + 0.3 * recent_trend_strength)
    
    return direction, final_confidence

def detect_patterns(candles, thresholds=None, enabled_patterns=None):
    """
    Обнаружение всех поддерживаемых паттернов на графике
    
    Args:
        candles: список обнаруженных свечей
        thresholds: словарь с порогами для разных паттернов
        enabled_patterns: список активных паттернов (если None, проверяются все)
        
    Returns:
        str: название обнаруженного паттерна или None
        float: уверенность в паттерне (0.0-1.0)
        dict: дополнительная информация о паттерне
    """
    try:
        # Значения по умолчанию для порогов
        default_thresholds = {
            "Двойное дно": 0.2,
            "Двойная вершина": 0.2,
            "Восходящий клин": 0.25,
            "Нисходящий клин": 0.25,
            "Линия тренда": 0.2
        }
        
        # Используем предоставленные пороги или значения по умолчанию
        if thresholds is None:
            thresholds = default_thresholds
        else:
            # Объединяем предоставленные пороги с дефолтными
            for key, value in default_thresholds.items():
                if key not in thresholds:
                    thresholds[key] = value
        
        if len(candles) < 7:  # Минимальное количество свечей для анализа
            return None, 0.0, {}
            
        # Проверка всех паттернов
        pattern_results = []
        
        # Двойное дно
        if enabled_patterns is None or "Двойное дно" in enabled_patterns:
            is_double_bottom, confidence_db = detect_double_bottom(candles, thresholds["Двойное дно"])
            if is_double_bottom:
                pattern_results.append(("Двойное дно", confidence_db, {}))
        
        # Двойная вершина
        if enabled_patterns is None or "Двойная вершина" in enabled_patterns:
            is_double_top, confidence_dt = detect_double_top(candles, thresholds["Двойная вершина"])
            if is_double_top:
                pattern_results.append(("Двойная вершина", confidence_dt, {}))
        
        # Восходящий клин
        if enabled_patterns is None or "Восходящий клин" in enabled_patterns:
            is_ascending_wedge, confidence_aw = detect_ascending_wedge(candles, threshold=thresholds["Восходящий клин"])
            if is_ascending_wedge:
                pattern_results.append(("Восходящий клин", confidence_aw, {}))
        
        # Нисходящий клин
        if enabled_patterns is None or "Нисходящий клин" in enabled_patterns:
            is_descending_wedge, confidence_dw = detect_descending_wedge(candles, threshold=thresholds["Нисходящий клин"])
            if is_descending_wedge:
                pattern_results.append(("Нисходящий клин", confidence_dw, {}))
        
        # Линия тренда
        if enabled_patterns is None or "Линия тренда" in enabled_patterns:
            is_trendline, confidence_tl, trendline_info = detect_trendline(candles, threshold=thresholds["Линия тренда"])
            if is_trendline:
                pattern_results.append(("Линия тренда", confidence_tl, trendline_info))
        
        # Если обнаружены паттерны, выбираем с наивысшей уверенностью
        if pattern_results:
            best_pattern = max(pattern_results, key=lambda x: x[1])
            return best_pattern[0], best_pattern[1], best_pattern[2]
        
        return None, 0.0, {}
    
    except Exception as e:
        logger.error(f"Ошибка при обнаружении паттернов: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, 0.0, {}