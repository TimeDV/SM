"""
Индикаторы для анализа бинарных опционов в ScalpMaster (SM)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_rsi(candles, period=14):
    """
    Расчет индикатора RSI (Relative Strength Index)
    
    Args:
        candles: список свечей с информацией о цене
        period: период для расчета RSI
        
    Returns:
        float: значение RSI (0-100)
    """
    if len(candles) < period + 1:
        return 50  # Недостаточно данных
    
    # Извлекаем цены закрытия
    closes = [c["center_y"] for c in candles[-period-1:]]
    
    # Инвертируем значения Y, так как на экране Y растет вниз
    closes = [-c for c in closes]
    
    # Рассчитываем изменения цены
    deltas = np.diff(closes)
    
    # Разделяем положительные и отрицательные изменения
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Рассчитываем средние значения
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def detect_bollinger_touch(candles, period=20, std_dev=2):
    """
    Определение касания полос Боллинджера
    
    Args:
        candles: список свечей с информацией о цене
        period: период для расчета полос Боллинджера
        std_dev: количество стандартных отклонений
        
    Returns:
        str: "upper" (касание верхней полосы), "lower" (касание нижней полосы) или None
    """
    if len(candles) < period:
        return None
    
    # Извлекаем цены закрытия (инвертируем Y)
    closes = [-c["center_y"] for c in candles[-period:]]
    
    # Рассчитываем среднюю и стандартное отклонение
    mean = np.mean(closes)
    sigma = np.std(closes)
    
    # Рассчитываем полосы Боллинджера
    upper_band = mean + std_dev * sigma
    lower_band = mean - std_dev * sigma
    
    # Инвертируем обратно для сравнения (на экране Y растет вниз)
    upper_band = -upper_band
    lower_band = -lower_band
    
    # Проверяем касание последней свечой
    last_candle = candles[-1]
    
    if last_candle["top"] <= upper_band and last_candle["top"] >= upper_band * 0.995:
        return "upper"
    elif last_candle["bottom"] >= lower_band and last_candle["bottom"] <= lower_band * 1.005:
        return "lower"
    
    return None

def detect_momentum_signal(candles, period=10):
    """
    Определение сигнала моментума
    
    Args:
        candles: список свечей с информацией о цене
        period: период для расчета моментума
        
    Returns:
        tuple: (направление, сила сигнала)
    """
    if len(candles) < period + 5:
        return None, 0
    
    # Извлекаем цены закрытия (инвертируем Y, т.к. на экране Y растет вниз)
    closes = [-c["center_y"] for c in candles[-(period+5):]]
    
    # Рассчитываем моментум (скорость изменения цены)
    momentum = closes[-1] - closes[-period]
    
    # Рассчитываем среднюю скорость изменения за 5 свечей
    recent_momentum = closes[-1] - closes[-5]
    
    # Определяем направление и силу сигнала
    if momentum < 0 and recent_momentum < 0:
        # Нисходящий моментум
        strength = min(abs(recent_momentum / momentum) * 100, 100) if momentum != 0 else 50
        return "down", strength
    elif momentum > 0 and recent_momentum > 0:
        # Восходящий моментум
        strength = min(abs(recent_momentum / momentum) * 100, 100) if momentum != 0 else 50
        return "up", strength
    
    # Неоднозначный моментум
    return None, 0

def detect_pin_bar(candles):
    """
    Обнаружение паттерна "Пин-бар" (разворотная свеча с длинным хвостом)
    
    Args:
        candles: список свечей с информацией о цене
        
    Returns:
        tuple: (направление, уверенность)
    """
    if len(candles) < 3:
        return None, 0
    
    last_candle = candles[-1]
    body_height = abs(last_candle["top"] - last_candle["bottom"])
    total_height = last_candle["height"]
    
    # Проверяем соотношение тела к хвосту
    if body_height <= total_height * 0.3:
        # Определяем направление по положению тела свечи
        if last_candle["top"] < (last_candle["bottom"] + last_candle["top"]) / 2:
            # Пин-бар с верхним хвостом - сигнал вниз
            confidence = min((total_height / body_height) * 0.2, 0.95) if body_height > 0 else 0.7
            return "down", confidence
        else:
            # Пин-бар с нижним хвостом - сигнал вверх
            confidence = min((total_height / body_height) * 0.2, 0.95) if body_height > 0 else 0.7
            return "up", confidence
    
    return None, 0

def detect_engulfing_pattern(candles):
    """
    Обнаружение паттерна "Поглощение"
    
    Args:
        candles: список свечей с информацией о цене
        
    Returns:
        tuple: (направление, уверенность)
    """
    if len(candles) < 2:
        return None, 0
    
    prev_candle = candles[-2]
    curr_candle = candles[-1]
    
    prev_body_size = abs(prev_candle["top"] - prev_candle["bottom"])
    curr_body_size = abs(curr_candle["top"] - curr_candle["bottom"])
    
    # Проверка на бычье поглощение
    if (prev_candle["color"] == "red" and curr_candle["color"] == "green" and
        curr_body_size > prev_body_size and
        curr_candle["top"] < prev_candle["top"] and
        curr_candle["bottom"] > prev_candle["bottom"]):
        confidence = min(0.6 + (curr_body_size / prev_body_size) * 0.3, 0.95)
        return "up", confidence
    
    # Проверка на медвежье поглощение
    elif (prev_candle["color"] == "green" and curr_candle["color"] == "red" and
          curr_body_size > prev_body_size and
          curr_candle["top"] < prev_candle["top"] and
          curr_candle["bottom"] > prev_candle["bottom"]):
        confidence = min(0.6 + (curr_body_size / prev_body_size) * 0.3, 0.95)
        return "down", confidence
    
    return None, 0

def detect_false_breakout(candles, period=20):
    """
    Обнаружение ложного пробоя (фейк-брейк)
    
    Args:
        candles: список свечей с информацией о цене
        period: период для определения уровней
        
    Returns:
        tuple: (направление, уверенность)
    """
    if len(candles) < period + 3:
        return None, 0
    
    # Находим локальные максимумы и минимумы
    highs = [c["top"] for c in candles[-period-3:-3]]
    lows = [c["bottom"] for c in candles[-period-3:-3]]
    
    resistance_level = min(highs)  # Верхний уровень (на экране Y растет вниз)
    support_level = max(lows)      # Нижний уровень (на экране Y растет вниз)
    
    # Последние 3 свечи
    last_candles = candles[-3:]
    
    # Проверка на ложный пробой уровня сопротивления (вверх)
    if (any(c["top"] < resistance_level * 0.99 for c in last_candles) and
        last_candles[-1]["top"] > resistance_level):
        distance = abs(last_candles[-2]["top"] - resistance_level) / resistance_level
        confidence = min(0.7 + distance * 30, 0.95)
        return "down", confidence
    
    # Проверка на ложный пробой уровня поддержки (вниз)
    elif (any(c["bottom"] > support_level * 1.01 for c in last_candles) and
          last_candles[-1]["bottom"] < support_level):
        distance = abs(last_candles[-2]["bottom"] - support_level) / support_level
        confidence = min(0.7 + distance * 30, 0.95)
        return "up", confidence
    
    return None, 0
    
def detect_trendline_signal(candles, min_points=6):
        '''
        Обнаружение сигнала на основе линии тренда для бинарных опционов
        
        Args:
            candles: список свечей
            min_points: минимальное количество точек для определения тренда
            
        Returns:
            tuple: (направление, уверенность)
        '''
        from trendline_pattern import detect_trendline
        
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