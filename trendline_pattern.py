"""
Исправленный модуль для обнаружения паттерна "Линия тренда" в системе ScalpMaster
"""

import numpy as np
import cv2
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

def detect_trendline(candles, min_points=4, min_quality=0.7, min_angle=5, max_angle=45, threshold=0.2):
    """
    Обнаружение паттерна "Линия тренда"
    """
    try:
        if len(candles) < min_points:
            return False, 0.0, {}
        
        # Используем последние N свечей для определения тренда
        recent_candles = candles[-min_points:]
        
        # Извлекаем координаты центров свечей
        x_coords = np.array([c["center_x"] for c in recent_candles])
        y_coords = np.array([c["center_y"] for c in recent_candles])
        
        # Линейная регрессия для определения тренда - исправление
        # Используем стандартную формулу линейной регрессии без full=True/False
        coeffs = np.polyfit(x_coords, y_coords, 1)
        slope, intercept = coeffs
        
        # Вычисляем r_squared (коэффициент детерминации)
        # Прогнозируемые значения y
        y_pred = slope * x_coords + intercept
        # Среднее значение y
        y_mean = np.mean(y_coords)
        # Сумма квадратов общая
        ss_tot = np.sum((y_coords - y_mean) ** 2)
        # Сумма квадратов остатков
        ss_res = np.sum((y_coords - y_pred) ** 2)
        # Коэффициент детерминации
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Вычисляем угол наклона линии тренда (в градусах)
        angle_degrees = abs(np.degrees(np.arctan(slope)))
        
        # Проверяем критерии для определения линии тренда
        if r_squared < min_quality:
            return False, 0.0, {}
            
        if angle_degrees < min_angle or angle_degrees > max_angle:
            return False, 0.0, {}
        
        # Определяем направление тренда
        # Помним, что на экране ось Y направлена вниз
        if slope > 0:
            direction = "down"  # Нисходящий тренд (Y растет при увеличении X)
        else:
            direction = "up"    # Восходящий тренд (Y уменьшается при увеличении X)
        
        # Вычисляем уверенность в тренде
        # На основе качества аппроксимации и угла наклона
        confidence_quality = min(1.0, float(r_squared) / min_quality)  # ИСПРАВЛЕНИЕ: добавляем float()
        
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
        
        # Проверка точек касания - минимум 3 свечи должны быть близки к линии тренда
        touches = 0
        max_distance = 0
        
        for candle in recent_candles:
            # Расчет ожидаемого значения Y на линии тренда для данного X
            expected_y = float(slope) * candle["center_x"] + float(intercept)  # ИСПРАВЛЕНИЕ: добавляем float()
            
            # Расстояние от центра свечи до линии тренда
            distance = abs(candle["center_y"] - expected_y)
            max_distance = max(max_distance, distance)
            
            # Если свеча находится близко к линии тренда, считаем это касанием
            if distance < candle["height"] * 0.3:  # Допуск 30% от высоты свечи
                touches += 1
        
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
        
        # Если цвета свечей не соответствуют направлению тренда, снижаем уверенность
        if not color_match:
            confidence *= 0.8
        
        # Дополнительная информация о тренде
        trendline_info = {
            "direction": direction,
            "slope": float(slope),  # ИСПРАВЛЕНИЕ: добавляем float()
            "intercept": float(intercept),  # ИСПРАВЛЕНИЕ: добавляем float()
            "r_squared": float(r_squared),  # ИСПРАВЛЕНИЕ: добавляем float()
            "angle_degrees": float(angle_degrees),  # ИСПРАВЛЕНИЕ: добавляем float()
            "points": len(recent_candles),
            "touches": touches,
            "is_color_match": color_match
        }
        
        # Применяем порог чувствительности (если указан)
        if threshold > 0 and confidence < threshold:
            return False, 0.0, {}
        
        return True, min(0.95, confidence), trendline_info
    
    except Exception as e:
        print(f"Ошибка при обнаружении линии тренда: {e}")
        return False, 0.0, {}

# Остальные функции оставляем без изменений
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
        print(f"Ошибка при отрисовке линии тренда: {e}")

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
    # Проверим последние 3 свечи на соответствие направлению тренда
    
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