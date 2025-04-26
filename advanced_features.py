"""
Модуль расширенных признаков для ScalpMaster (SM)
Содержит функции для вычисления дополнительных признаков, рекомендованных DeepSeek
"""

import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
import pytz
from scipy.signal import butter, filtfilt

# Настройка логирования
logger = logging.getLogger("advanced_features")
if not logger.handlers:
    file_handler = logging.FileHandler("sm_advanced_features.log", encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


class AdvancedFeatureExtractor:
    def __init__(self):
        """
        Инициализация экстрактора расширенных признаков
        """
        # Сессии и их время (UTC)
        self.market_sessions = {
            "asia": {"start": 22, "end": 8},  # 22:00-8:00 UTC
            "europe": {"start": 7, "end": 16},  # 7:00-16:00 UTC
            "us": {"start": 13, "end": 22}  # 13:00-22:00 UTC
        }
        
        # Кэш для избежания повторных вычислений
        self.feature_cache = {}
        self.cache_ttl = 60  # Время жизни кэша в секундах
        self.cache_timestamp = time.time()
        
        logger.info("AdvancedFeatureExtractor инициализирован")
    
    def clear_cache(self):
        """Очистка кэша признаков"""
        self.feature_cache = {}
        self.cache_timestamp = time.time()
        logger.info("Кэш признаков очищен")
    
    def _is_cache_valid(self):
        """Проверка действительности кэша"""
        return time.time() - self.cache_timestamp < self.cache_ttl
    
    def _butter_bandpass_filter(self, data, lowcut=0.01, highcut=0.5, fs=1.0, order=5):
        """
        Реализация полосового фильтра Баттерворта
        
        Args:
            data: массив данных для фильтрации
            lowcut: нижняя частота среза
            highcut: верхняя частота среза
            fs: частота дискретизации
            order: порядок фильтра
        
        Returns:
            np.array: отфильтрованные данные
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def _get_current_session(self):
        """
        Определение текущей торговой сессии на основе времени UTC
        
        Returns:
            str: название активной сессии или "overlap" при пересечении сессий
        """
        current_hour = datetime.now(pytz.UTC).hour
        active_sessions = []
        
        for session, times in self.market_sessions.items():
            start = times["start"]
            end = times["end"]
            
            # Обрабатываем переход через полночь
            if start < end:
                if start <= current_hour < end:
                    active_sessions.append(session)
            else:  # start > end (переход через полночь)
                if start <= current_hour or current_hour < end:
                    active_sessions.append(session)
        
        if len(active_sessions) == 0:
            return "closed"
        elif len(active_sessions) == 1:
            return active_sessions[0]
        else:
            return "overlap"
    
    def extract_price_velocity(self, candles, period=5):
        """
        Вычисление скорости изменения цены: (Close[-1] - Close[-period]) / Close[-period]
        
        Args:
            candles: список свечей
            period: период для расчета скорости (кол-во свечей)
            
        Returns:
            float: скорость изменения цены
        """
        if len(candles) < period + 1:
            return 0.0
        
        # На экране Y растет вниз, поэтому инвертируем для обычной логики
        current_price = -candles[-1]["center_y"]
        past_price = -candles[-period-1]["center_y"]
        
        if past_price == 0:
            return 0.0
        
        return (current_price - past_price) / abs(past_price)
    
    def extract_filtered_price(self, candles):
        """
        Применение фильтра Баттерворта для сглаживания ценового ряда
        
        Args:
            candles: список свечей
            
        Returns:
            dict: отфильтрованные данные {
                "filtered_prices": список отфильтрованных цен,
                "noise_level": уровень шума (стандартное отклонение разности)
            }
        """
        if len(candles) < 10:
            return {"filtered_prices": [], "noise_level": 0.0}
        
        # Извлекаем цены (инвертируем Y для обычной логики)
        prices = [-c["center_y"] for c in candles]
        
        # Применяем фильтр
        filtered = self._butter_bandpass_filter(prices)
        
        # Вычисляем уровень шума как стандартное отклонение разности
        noise = np.std(np.array(prices) - filtered)
        
        return {
            "filtered_prices": filtered.tolist(),
            "noise_level": float(noise)
        }
    
    def extract_session_features(self):
        """
        Извлечение признаков, связанных с торговыми сессиями
        
        Returns:
            dict: признаки сессий в формате one-hot encoding
        """
        current_session = self._get_current_session()
        
        # One-hot encoding для сессий
        return {
            "session_asia": 1 if current_session == "asia" else 0,
            "session_europe": 1 if current_session == "europe" else 0,
            "session_us": 1 if current_session == "us" else 0,
            "session_overlap": 1 if current_session == "overlap" else 0,
            "session_closed": 1 if current_session == "closed" else 0
        }
    
    def calculate_advanced_features(self, candles, pair=None):
        """
        Вычисление всех расширенных признаков для заданных свечей
        
        Args:
            candles: список свечей
            pair: валютная пара (опционально)
            
        Returns:
            dict: расширенные признаки
        """
        # Проверка валидности кэша
        cache_key = str(id(candles))
        if cache_key in self.feature_cache and self._is_cache_valid():
            logger.debug("Использование кэшированных признаков")
            return self.feature_cache[cache_key]
        
        try:
            if len(candles) < 10:
                logger.warning(f"Недостаточно свечей для расчета расширенных признаков: {len(candles)} < 10")
                return {}
            
            # Вычисляем все признаки
            features = {}
            
            # 1. Скорость изменения цены за разные периоды
            features["price_velocity_5"] = self.extract_price_velocity(candles, period=5)
            features["price_velocity_10"] = self.extract_price_velocity(candles, period=10)
            
            # 2. Фильтрация шума
            filtered_data = self.extract_filtered_price(candles)
            features["noise_level"] = filtered_data["noise_level"]
            
            # 3. Признаки торговой сессии
            session_features = self.extract_session_features()
            features.update(session_features)
            
            # 4. Признаки волатильности
            heights = [c["height"] for c in candles[-10:]]
            features["recent_volatility"] = np.std(heights)
            features["avg_candle_height"] = np.mean(heights)
            
            # 5. Признаки дисбаланса свечей
            recent_candles = candles[-10:]
            green_count = sum(1 for c in recent_candles if c["color"] == "green")
            red_count = sum(1 for c in recent_candles if c["color"] == "red")
            features["green_red_ratio"] = green_count / max(red_count, 1)
            
            # 6. Признаки импульса
            if len(candles) >= 20:
                # Импульс - изменение средней цены за последние N свечей
                recent_mean = np.mean([-c["center_y"] for c in candles[-10:]])
                previous_mean = np.mean([-c["center_y"] for c in candles[-20:-10]])
                features["momentum"] = (recent_mean - previous_mean) / abs(previous_mean) if previous_mean != 0 else 0
            
            # 7. Признаки неравномерности движения
            if len(candles) >= 5:
                moves = []
                for i in range(1, 5):
                    # Процентное изменение между соседними свечами
                    curr = -candles[-i]["center_y"]
                    prev = -candles[-i-1]["center_y"]
                    moves.append((curr - prev) / abs(prev) if prev != 0 else 0)
                features["move_irregularity"] = np.std(moves)
            
            # Кэшируем результаты
            self.feature_cache[cache_key] = features
            self.cache_timestamp = time.time()
            
            return features
        
        except Exception as e:
            logger.error(f"Ошибка при вычислении расширенных признаков: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}