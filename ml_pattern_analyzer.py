"""
ML Pattern Analyzer для системы ScalpMaster (SM)
Модуль машинного обучения для улучшения распознавания паттернов
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
import json
from datetime import datetime
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Настройка логирования
logger = logging.getLogger("ml_pattern_analyzer")
if not logger.handlers:
    file_handler = logging.FileHandler("sm_ml_analyzer.log", encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

class MLPatternAnalyzer:
    def __init__(self, model_dir="ml_models"):
        """
        Инициализация анализатора паттернов на основе машинного обучения
        
        Args:
            model_dir: директория для хранения моделей
        """
        self.model_dir = model_dir
        self.models = {}  # словарь моделей для разных паттернов
        self.scalers = {}  # словарь нормализаторов данных для разных паттернов
        self.features = []  # список используемых признаков
        self.pattern_names = ["Двойное дно", "Двойная вершина", "Восходящий клин", 
                             "Нисходящий клин", "Линия тренда"]
        
        # Создаем директорию для моделей, если ее нет
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Загружаем существующие модели, если они есть
        self._load_models()
        
        logger.info("ML Pattern Analyzer инициализирован")
    
    def _load_models(self):
        """Загрузка сохраненных моделей из директории model_dir"""
        try:
            for pattern in self.pattern_names:
                model_path = os.path.join(self.model_dir, f"{pattern.lower().replace(' ', '_')}_model.pkl")
                scaler_path = os.path.join(self.model_dir, f"{pattern.lower().replace(' ', '_')}_scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    with open(model_path, 'rb') as f:
                        self.models[pattern] = pickle.load(f)
                    
                    with open(scaler_path, 'rb') as f:
                        self.scalers[pattern] = pickle.load(f)
                    
                    logger.info(f"Модель для паттерна '{pattern}' успешно загружена")
            
            # Загружаем список признаков
            features_path = os.path.join(self.model_dir, "features.json")
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.features = json.load(f)
                logger.info(f"Загружен список признаков: {len(self.features)} признаков")
        
        except Exception as e:
            logger.error(f"Ошибка при загрузке моделей: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _save_model(self, pattern, model, scaler):
        """
        Сохранение обученной модели и нормализатора
        
        Args:
            pattern: название паттерна
            model: обученная модель
            scaler: нормализатор данных
        """
        try:
            model_path = os.path.join(self.model_dir, f"{pattern.lower().replace(' ', '_')}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{pattern.lower().replace(' ', '_')}_scaler.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Сохраняем список признаков
            features_path = os.path.join(self.model_dir, "features.json")
            with open(features_path, 'w') as f:
                json.dump(self.features, f)
            
            logger.info(f"Модель для паттерна '{pattern}' успешно сохранена")
            return True
        
        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def extract_features(self, candles):
        """
        Извлечение признаков из списка свечей для машинного обучения
        
        Args:
            candles: список обнаруженных свечей
            
        Returns:
            dict: словарь признаков
        """
        if len(candles) < 5:
            logger.warning("Недостаточно свечей для извлечения признаков")
            return {}
        
        # Создаем DataFrame для удобства анализа
        df = pd.DataFrame([{
            'x': c['center_x'],
            'y': c['center_y'],
            'height': c['height'],
            'width': c['width'],
            'top': c['top'],
            'bottom': c['bottom'],
            'color': c['color']
        } for c in candles])
        
        # Извлекаем базовые статистические признаки
        features = {}
        
        # 1. Признаки высот свечей
        features['mean_height'] = df['height'].mean()
        features['std_height'] = df['height'].std()
        features['max_height'] = df['height'].max()
        features['min_height'] = df['height'].min()
        
        # 2. Признаки положения свечей
        features['mean_y'] = df['y'].mean()
        features['std_y'] = df['y'].std()
        features['y_range'] = df['y'].max() - df['y'].min()
        
        # 3. Анализ цветов свечей
        features['green_ratio'] = (df['color'] == 'green').mean()
        features['red_ratio'] = (df['color'] == 'red').mean()
        
        # 4. Признаки тренда (линейная регрессия)
        x = np.array(df['x'])
        y = np.array(df['y'])
        
        if len(x) > 1:  # Проверка, достаточно ли точек для регрессии
            slope, intercept = np.polyfit(x, y, 1)
            features['trend_slope'] = slope
            # На экране Y растет вниз, поэтому negative_slope означает восходящий тренд
            features['is_uptrend'] = 1 if slope < 0 else 0
            
            # Вычисляем R^2 (коэффициент детерминации)
            y_pred = slope * x + intercept
            y_mean = np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            features['trend_r2'] = r_squared
        else:
            features['trend_slope'] = 0
            features['is_uptrend'] = 0
            features['trend_r2'] = 0
        
        # 5. Поиск локальных экстремумов
        tops = []
        bottoms = []
        
        for i in range(1, len(df) - 1):
            # Локальный максимум
            if df.iloc[i]['top'] < df.iloc[i-1]['top'] and df.iloc[i]['top'] < df.iloc[i+1]['top']:
                tops.append(i)
            
            # Локальный минимум
            if df.iloc[i]['bottom'] > df.iloc[i-1]['bottom'] and df.iloc[i]['bottom'] > df.iloc[i+1]['bottom']:
                bottoms.append(i)
        
        features['num_tops'] = len(tops)
        features['num_bottoms'] = len(bottoms)
        
        # 6. Если есть два или более минимумов/максимумов, анализируем их
        if len(bottoms) >= 2:
            # Расстояние между последними двумя минимумами
            features['last_bottoms_distance'] = bottoms[-1] - bottoms[-2]
            
            # Разница по Y между последними двумя минимумами
            if len(bottoms) >= 2:
                y1 = df.iloc[bottoms[-2]]['bottom']
                y2 = df.iloc[bottoms[-1]]['bottom']
                features['last_bottoms_y_diff'] = abs(y1 - y2)
                features['last_bottoms_y_diff_ratio'] = abs(y1 - y2) / features['y_range'] if features['y_range'] > 0 else 0
        else:
            features['last_bottoms_distance'] = 0
            features['last_bottoms_y_diff'] = 0
            features['last_bottoms_y_diff_ratio'] = 0
        
        if len(tops) >= 2:
            # Расстояние между последними двумя максимумами
            features['last_tops_distance'] = tops[-1] - tops[-2]
            
            # Разница по Y между последними двумя максимумами
            if len(tops) >= 2:
                y1 = df.iloc[tops[-2]]['top']
                y2 = df.iloc[tops[-1]]['top']
                features['last_tops_y_diff'] = abs(y1 - y2)
                features['last_tops_y_diff_ratio'] = abs(y1 - y2) / features['y_range'] if features['y_range'] > 0 else 0
        else:
            features['last_tops_distance'] = 0
            features['last_tops_y_diff'] = 0
            features['last_tops_y_diff_ratio'] = 0
        
        # 7. Последовательные направления свечей
        directions = []
        for i in range(1, len(df)):
            prev_y = df.iloc[i-1]['y']
            curr_y = df.iloc[i]['y']
            directions.append(1 if curr_y < prev_y else -1 if curr_y > prev_y else 0)
        
        features['direction_changes'] = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
        
        # 8. Анализ последних N свечей
        last_n = min(5, len(df))
        features['last_n_green_ratio'] = (df.iloc[-last_n:]['color'] == 'green').mean()
        features['last_n_red_ratio'] = (df.iloc[-last_n:]['color'] == 'red').mean()
        
        return features
    
    def train_models(self, log_file="pattern_forecast_log.csv", test_size=0.2):
        """
        Обучение моделей машинного обучения на основе логов прогнозов
        
        Args:
            log_file: путь к CSV-файлу с логами прогнозов
            test_size: доля тестовых данных (0.0-1.0)
            
        Returns:
            dict: результаты обучения для каждого паттерна
        """
        if not os.path.exists(log_file):
            logger.error(f"Файл с логами {log_file} не найден")
            return {"status": "error", "message": f"Файл {log_file} не найден"}
        
        try:
            # Загружаем логи прогнозов
            df = pd.read_csv(log_file)
            logger.info(f"Загружено {len(df)} записей из лога прогнозов")
            
            # Проверяем наличие данных
            if len(df) < 50:
                logger.warning(f"Недостаточно данных для обучения ({len(df)} записей)")
                return {"status": "warning", "message": f"Недостаточно данных для обучения ({len(df)} записей)"}
            
            # Проверяем необходимые колонки
            required_columns = ['Паттерн', 'Правильно', 'Ключевые точки']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"В логе отсутствуют необходимые колонки: {', '.join(missing_columns)}")
                return {"status": "error", "message": f"В логе отсутствуют необходимые колонки: {', '.join(missing_columns)}"}
            
            # Обрабатываем колонку с ключевыми точками (конвертируем из JSON)
            df['Ключевые точки'] = df['Ключевые точки'].apply(lambda x: json.loads(x) if isinstance(x, str) and x.strip() else {})
            
            # Конвертируем ответы в бинарный вид
            df['Правильно'] = df['Правильно'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
            
            # Собираем все уникальные ключи из словарей в колонке "Ключевые точки"
            key_point_features = set()
            for kp_dict in df['Ключевые точки']:
                if isinstance(kp_dict, dict):
                    key_point_features.update(kp_dict.keys())
            
            # Создаем DataFrame с признаками
            features_df = pd.DataFrame()
            
            # Добавляем признаки из ключевых точек
            for feature in key_point_features:
                features_df[feature] = df['Ключевые точки'].apply(lambda x: x.get(feature, np.nan) if isinstance(x, dict) else np.nan)
            
            # Сохраняем список признаков
            self.features = list(features_df.columns)
            
            # Обучаем отдельную модель для каждого паттерна
            results = {}
            
            for pattern in self.pattern_names:
                pattern_df = df[df['Паттерн'] == pattern]
                
                if len(pattern_df) < 20:
                    logger.warning(f"Недостаточно данных для паттерна '{pattern}' ({len(pattern_df)} записей)")
                    results[pattern] = {"status": "warning", "message": f"Недостаточно данных ({len(pattern_df)} записей)"}
                    continue
                
                # Выбираем признаки и ответы для текущего паттерна
                pattern_features = features_df.loc[pattern_df.index]
                pattern_labels = pattern_df['Правильно'].values
                
                # Заполняем пропущенные значения
                pattern_features = pattern_features.fillna(0)
                
                # Разделяем на обучающую и тестовую выборки
                X_train, X_test, y_train, y_test = train_test_split(
                    pattern_features, pattern_labels, test_size=test_size, random_state=42
                )
                
                # Нормализуем данные
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Обучаем модель (используем RandomForest как надежный классификатор)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Оцениваем качество модели
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Сохраняем модель и нормализатор
                self.models[pattern] = model
                self.scalers[pattern] = scaler
                self._save_model(pattern, model, scaler)
                
                # Результаты
                results[pattern] = {
                    "status": "success",
                    "accuracy": accuracy,
                    "sample_size": len(pattern_df),
                    "feature_importance": dict(zip(pattern_features.columns, model.feature_importances_))
                }
                
                logger.info(f"Модель для паттерна '{pattern}' обучена, точность: {accuracy:.2f}, количество примеров: {len(pattern_df)}")
            
            return {"status": "success", "results": results}
            
        except Exception as e:
            logger.error(f"Ошибка при обучении моделей: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}
    
    def predict_pattern(self, pattern, features):
        """
        Оценка вероятности паттерна по признакам
        
        Args:
            pattern: название паттерна
            features: словарь признаков
            
        Returns:
            tuple: (вероятность паттерна, уверенность в правильности)
        """
        if pattern not in self.models or pattern not in self.scalers:
            logger.warning(f"Модель для паттерна '{pattern}' не найдена")
            return 0.0, 0.0
        
        try:
            # Подготовка признаков
            feature_vector = []
            for feature in self.features:
                feature_vector.append(features.get(feature, 0))
            
            # Конвертируем в numpy array
            X = np.array([feature_vector])
            
            # Нормализуем
            X_scaled = self.scalers[pattern].transform(X)
            
            # Получаем вероятности классов
            proba = self.models[pattern].predict_proba(X_scaled)[0]
            
            # Вероятность положительного класса (паттерн корректен)
            positive_proba = proba[1]
            
            # Оценка уверенности (насколько близко к 0 или 1)
            confidence = max(positive_proba, 1 - positive_proba)
            
            return positive_proba, confidence
            
        except Exception as e:
            logger.error(f"Ошибка при прогнозировании: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0, 0.0
    
    def analyze_candles(self, candles, conventional_results):
        """
        Анализ свечей с использованием ML и объединение с результатами обычных алгоритмов
        
        Args:
            candles: список обнаруженных свечей
            conventional_results: результаты обычных алгоритмов в формате 
                                  (название паттерна, уверенность, информация)
        
        Returns:
            tuple: (название паттерна, уверенность, информация, ml_confidence)
        """
        # Извлекаем признаки из свечей
        features = self.extract_features(candles)
        
        # Если признаки не извлечены или моделей нет, возвращаем результаты обычных алгоритмов
        if not features or not self.models:
            return conventional_results + (0.0,)
        
        pattern, confidence, info = conventional_results
        
        # Если паттерн не обнаружен, возвращаем результаты обычных алгоритмов
        if not pattern:
            return pattern, confidence, info, 0.0
        
        # Получаем оценку от ML модели
        ml_proba, ml_confidence = self.predict_pattern(pattern, features)
        
        # Объединяем оценки традиционного алгоритма и ML
        # Используем взвешенный подход
        combined_confidence = 0.7 * confidence + 0.3 * ml_proba
        
        # Дополняем информацию о паттерне
        if isinstance(info, dict):
            info["ml_probability"] = float(ml_proba)
            info["ml_confidence"] = float(ml_confidence)
            info["combined_confidence"] = float(combined_confidence)
        
        return pattern, combined_confidence, info, ml_proba
    
    def get_feedback(self, pattern, features, is_correct):
        """
        Обработка обратной связи для улучшения модели
        
        Args:
            pattern: название паттерна
            features: словарь признаков
            is_correct: был ли прогноз верным
        """
        # Сохраняем пример для будущего дообучения
        feedback_dir = os.path.join(self.model_dir, "feedback")
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Формируем имя файла с меткой времени
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_path = os.path.join(feedback_dir, f"{pattern.lower().replace(' ', '_')}_{timestamp}.json")
        
        # Сохраняем данные
        feedback_data = {
            "pattern": pattern,
            "features": features,
            "is_correct": is_correct,
            "timestamp": timestamp
        }
        
        with open(feedback_path, 'w') as f:
            json.dump(feedback_data, f)
        
        logger.info(f"Сохранена обратная связь для паттерна '{pattern}'")
