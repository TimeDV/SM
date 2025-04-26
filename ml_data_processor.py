"""
ML Data Processor для системы ScalpMaster (SM)
Модуль для подготовки данных, генерации признаков и визуализации результатов ML-модели
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Настройка логирования
logger = logging.getLogger("ml_data_processor")
if not logger.handlers:
    file_handler = logging.FileHandler("sm_ml_data.log", encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

class MLDataProcessor:
    def __init__(self, data_dir="ml_data"):
        """
        Инициализация процессора данных для ML
        
        Args:
            data_dir: директория для хранения подготовленных данных
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info("ML Data Processor инициализирован")
    
    def prepare_training_data(self, log_file="pattern_forecast_log.csv", save=True):
        """
        Подготовка данных для обучения из лог-файла прогнозов
        
        Args:
            log_file: путь к файлу логов
            save: сохранить подготовленные данные в файл
            
        Returns:
            dict: подготовленные данные для каждого паттерна
        """
        if not os.path.exists(log_file):
            logger.error(f"Файл {log_file} не найден")
            return None
        
        try:
            # Загружаем данные из лог-файла
            df = pd.read_csv(log_file)
            logger.info(f"Загружено {len(df)} записей из файла {log_file}")
            
            # Проверяем необходимые колонки
            required_columns = ['Паттерн', 'Правильно', 'Ключевые точки']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"В логе отсутствуют необходимые колонки: {', '.join(missing_columns)}")
                return None
            
            # Обрабатываем данные
            processed_data = {}
            
            # Обрабатываем колонку с ключевыми точками (конвертируем из JSON)
            df['Ключевые точки'] = df['Ключевые точки'].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else {}
            )
            
            # Конвертируем ответы в бинарный вид
            df['Правильно'] = df['Правильно'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
            
            # Обрабатываем каждый паттерн
            patterns = df['Паттерн'].unique()
            
            for pattern in patterns:
                pattern_df = df[df['Паттерн'] == pattern].copy()
                
                # Пропускаем паттерны с малым количеством примеров
                if len(pattern_df) < 20:
                    logger.warning(f"Недостаточно данных для паттерна '{pattern}' ({len(pattern_df)} записей)")
                    continue
                
                # Собираем признаки из ключевых точек
                key_point_features = set()
                for kp_dict in pattern_df['Ключевые точки']:
                    if isinstance(kp_dict, dict):
                        key_point_features.update(kp_dict.keys())
                
                # Создаем DataFrame с признаками
                features_df = pd.DataFrame(index=pattern_df.index)
                
                # Добавляем признаки из ключевых точек
                for feature in key_point_features:
                    features_df[feature] = pattern_df['Ключевые точки'].apply(
                        lambda x: x.get(feature, np.nan) if isinstance(x, dict) else np.nan
                    )
                
                # Заполняем пропущенные значения
                features_df = features_df.fillna(0)
                
                # Добавляем другие признаки, если они есть в логе
                additional_features = ['Уверенность', 'Направление', 'Валютная пара', 'Таймфрейм']
                for feature in additional_features:
                    if feature in pattern_df.columns:
                        if feature == 'Направление':
                            # Конвертируем направление в числовое представление
                            direction_map = {'Вверх': 1, 'Вниз': -1}
                            features_df['direction_numeric'] = pattern_df[feature].apply(
                                lambda x: direction_map.get(x, 0)
                            )
                        elif feature == 'Валютная пара':
                            # Создаем дамми-переменные для валютных пар
                            pairs = pd.get_dummies(pattern_df[feature], prefix='pair')
                            features_df = pd.concat([features_df, pairs], axis=1)
                        elif feature == 'Таймфрейм':
                            # Преобразуем таймфрейм в числовое значение
                            features_df['timeframe_numeric'] = pd.to_numeric(pattern_df[feature], errors='coerce').fillna(0)
                        else:
                            # Копируем числовые признаки как есть
                            features_df[feature] = pd.to_numeric(pattern_df[feature], errors='coerce').fillna(0)
                
                # Сохраняем подготовленные данные
                processed_data[pattern] = {
                    'features': features_df,
                    'labels': pattern_df['Правильно'].values,
                    'sample_size': len(pattern_df)
                }
                
                logger.info(f"Подготовлены данные для паттерна '{pattern}': {len(features_df)} записей, {len(features_df.columns)} признаков")
                
                # Сохраняем в файл, если требуется
                if save:
                    pattern_file = os.path.join(self.data_dir, f"{pattern.lower().replace(' ', '_')}_data.pkl")
                    with open(pattern_file, 'wb') as f:
                        pickle.dump(processed_data[pattern], f)
                    
                    logger.info(f"Данные для паттерна '{pattern}' сохранены в {pattern_file}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке данных: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def analyze_features_importance(self, model, feature_names, output_file=None):
        """
        Анализ важности признаков модели
        
        Args:
            model: обученная модель с атрибутом feature_importances_
            feature_names: названия признаков
            output_file: путь для сохранения графика
            
        Returns:
            pd.DataFrame: DataFrame с важностью признаков
        """
        try:
            # Проверяем наличие атрибута feature_importances_
            if not hasattr(model, 'feature_importances_'):
                logger.warning("Модель не поддерживает анализ важности признаков")
                return None
            
            # Создаем DataFrame с важностью признаков
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            })
            
            # Сортируем по убыванию важности
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Создаем визуализацию
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=importance_df.head(20))
            plt.title('Top 20 Feature Importance')
            plt.tight_layout()
            
            # Сохраняем график, если указан выходной файл
            if output_file:
                plt.savefig(output_file)
                logger.info(f"График важности признаков сохранен в {output_file}")
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Ошибка при анализе важности признаков: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def visualize_data_distribution(self, data, pattern_name, output_file=None):
        """
        Визуализация распределения данных для паттерна
        
        Args:
            data: данные паттерна (dict с ключами 'features' и 'labels')
            pattern_name: название паттерна
            output_file: путь для сохранения графика
        """
        try:
            features = data['features']
            labels = data['labels']
            
            # Используем PCA для уменьшения размерности
            pca = PCA(n_components=2)
            scaler = StandardScaler()
            
            # Нормализуем данные
            features_scaled = scaler.fit_transform(features)
            
            # Применяем PCA
            features_pca = pca.fit_transform(features_scaled)
            
            # Создаем DataFrame для визуализации
            viz_df = pd.DataFrame({
                'PC1': features_pca[:, 0],
                'PC2': features_pca[:, 1],
                'label': labels
            })
            
            # Создаем визуализацию
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='PC1', y='PC2', hue='label', data=viz_df, palette={0: 'red', 1: 'green'})
            plt.title(f'PCA visualization for {pattern_name}')
            plt.legend(['Incorrect', 'Correct'])
            
            # Добавляем информацию о объясненной дисперсии
            explained_variance = pca.explained_variance_ratio_
            plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
            
            # Сохраняем график, если указан выходной файл
            if output_file:
                plt.savefig(output_file)
                logger.info(f"График распределения данных сохранен в {output_file}")
            
            # Дополнительно используем t-SNE для более сложных данных
            tsne = TSNE(n_components=2, random_state=42)
            features_tsne = tsne.fit_transform(features_scaled)
            
            # Создаем DataFrame для визуализации
            viz_df_tsne = pd.DataFrame({
                'tSNE1': features_tsne[:, 0],
                'tSNE2': features_tsne[:, 1],
                'label': labels
            })
            
            # Создаем визуализацию t-SNE
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='tSNE1', y='tSNE2', hue='label', data=viz_df_tsne, palette={0: 'red', 1: 'green'})
            plt.title(f't-SNE visualization for {pattern_name}')
            plt.legend(['Incorrect', 'Correct'])
            
            # Сохраняем график t-SNE, если указан выходной файл
            if output_file:
                tsne_output = output_file.replace('.png', '_tsne.png')
                plt.savefig(tsne_output)
                logger.info(f"График t-SNE распределения данных сохранен в {tsne_output}")
            
        except Exception as e:
            logger.error(f"Ошибка при визуализации распределения данных: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def plot_model_performance(self, model, X_test, y_test, pattern_name, output_dir=None):
        """
        Построение графиков производительности модели
        
        Args:
            model: обученная модель
            X_test: тестовые признаки
            y_test: тестовые метки
            pattern_name: название паттерна
            output_dir: директория для сохранения графиков
        """
        try:
            # Создаем директорию для графиков, если ее нет
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Получаем предсказания модели
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # 1. Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {pattern_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            if output_dir:
                cm_file = os.path.join(output_dir, f"{pattern_name.lower().replace(' ', '_')}_confusion_matrix.png")
                plt.savefig(cm_file)
                logger.info(f"График матрицы ошибок сохранен в {cm_file}")
            plt.close()
            
            # 2. ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {pattern_name}')
            plt.legend(loc='lower right')
            
            if output_dir:
                roc_file = os.path.join(output_dir, f"{pattern_name.lower().replace(' ', '_')}_roc_curve.png")
                plt.savefig(roc_file)
                logger.info(f"График ROC-кривой сохранен в {roc_file}")
            plt.close()
            
            # 3. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {pattern_name}')
            plt.legend(loc='lower left')
            
            if output_dir:
                pr_file = os.path.join(output_dir, f"{pattern_name.lower().replace(' ', '_')}_pr_curve.png")
                plt.savefig(pr_file)
                logger.info(f"График Precision-Recall сохранен в {pr_file}")
            plt.close()
            
            # 4. Результаты в виде справочной таблицы
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            
            results = {
                'Pattern': pattern_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC AUC': roc_auc,
                'PR AUC': pr_auc
            }
            
            if output_dir:
                results_file = os.path.join(output_dir, f"{pattern_name.lower().replace(' ', '_')}_results.json")
                with open(results_file, 'w') as f:
                    json.dump(results, f)
                logger.info(f"Результаты сохранены в {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при построении графиков производительности: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def generate_additional_features(self, candles):
        """
        Генерация дополнительных признаков из свечей
        
        Args:
            candles: список обнаруженных свечей
            
        Returns:
            dict: расширенный набор признаков
        """
        try:
            if len(candles) < 5:
                logger.warning("Недостаточно свечей для генерации признаков")
                return {}
            
            # Базовые признаки из свечей
            heights = [c["height"] for c in candles]
            centers = [c["center_y"] for c in candles]
            colors = [c["color"] for c in candles]
            tops = [c["top"] for c in candles]
            bottoms = [c["bottom"] for c in candles]
            
            # 1. Статистические признаки
            features = {
                "mean_height": np.mean(heights),
                "std_height": np.std(heights),
                "max_height": max(heights),
                "min_height": min(heights),
                "height_range": max(heights) - min(heights),
                
                "mean_center": np.mean(centers),
                "std_center": np.std(centers),
                "center_range": max(centers) - min(centers),
                
                "green_ratio": colors.count("green") / len(colors),
                "red_ratio": colors.count("red") / len(colors),
                
                "num_candles": len(candles)
            }
            
            # 2. Признаки тренда
            x = np.arange(len(candles))
            
            # Линейная регрессия для центров свечей
            slope_center, intercept_center = np.polyfit(x, centers, 1)
            features["trend_slope_center"] = slope_center
            features["trend_direction"] = "up" if slope_center < 0 else "down"  # Помним, что на экране Y растет вниз
            
            # Линейная регрессия для верхних и нижних границ
            slope_top, intercept_top = np.polyfit(x, tops, 1)
            slope_bottom, intercept_bottom = np.polyfit(x, bottoms, 1)
            
            features["trend_slope_top"] = slope_top
            features["trend_slope_bottom"] = slope_bottom
            
            # Вычисляем R^2 для центров свечей
            y_pred = slope_center * x + intercept_center
            y_mean = np.mean(centers)
            ss_tot = np.sum((centers - y_mean) ** 2)
            ss_res = np.sum((centers - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            features["trend_r2"] = r_squared
            features["trend_strength"] = abs(slope_center) * r_squared
            
            # 3. Признаки паттернов свечей
            # Считаем последовательные свечи одного цвета
            consecutive_green = 0
            consecutive_red = 0
            max_consecutive_green = 0
            max_consecutive_red = 0
            
            for color in colors:
                if color == "green":
                    consecutive_green += 1
                    consecutive_red = 0
                elif color == "red":
                    consecutive_red += 1
                    consecutive_green = 0
                else:
                    consecutive_green = 0
                    consecutive_red = 0
                
                max_consecutive_green = max(max_consecutive_green, consecutive_green)
                max_consecutive_red = max(max_consecutive_red, consecutive_red)
            
            features["max_consecutive_green"] = max_consecutive_green
            features["max_consecutive_red"] = max_consecutive_red
            
            # 4. Анализ изменений направления
            direction_changes = 0
            for i in range(1, len(candles)):
                prev_center = candles[i-1]["center_y"]
                curr_center = candles[i]["center_y"]
                
                if (prev_center < curr_center and i > 1 and candles[i-2]["center_y"] > prev_center) or \
                   (prev_center > curr_center and i > 1 and candles[i-2]["center_y"] < prev_center):
                    direction_changes += 1
            
            features["direction_changes"] = direction_changes
            features["direction_change_ratio"] = direction_changes / (len(candles) - 1) if len(candles) > 1 else 0
            
            # 5. Признаки для оценки волатильности
            body_sizes = [abs(c["top"] - c["bottom"]) for c in candles]
            features["mean_body_size"] = np.mean(body_sizes)
            features["std_body_size"] = np.std(body_sizes)
            
            # 6. Анализ последних свечей
            last_n = min(3, len(candles))
            last_candles = candles[-last_n:]
            
            features["last_n_green_ratio"] = sum(1 for c in last_candles if c["color"] == "green") / last_n
            features["last_n_red_ratio"] = sum(1 for c in last_candles if c["color"] == "red") / last_n
            
            # Соответствие направления последних свечей общему тренду
            last_slope = 0
            if last_n > 1:
                last_x = np.arange(last_n)
                last_centers = [c["center_y"] for c in last_candles]
                last_slope, _ = np.polyfit(last_x, last_centers, 1)
            
            features["last_candles_slope"] = last_slope
            features["last_candles_trend_match"] = (last_slope * slope_center < 0)  # True если последние свечи противоречат общему тренду
            
            return features
            
        except Exception as e:
            logger.error(f"Ошибка при генерации дополнительных признаков: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}