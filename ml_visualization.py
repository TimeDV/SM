"""
Модуль визуализации результатов ML для ScalpMaster (SM)
Содержит функции для графического представления прогнозов ML и LSTM
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import os
import time
from datetime import datetime, timedelta
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Настройка логирования
logger = logging.getLogger("ml_visualization")
if not logger.handlers:
    file_handler = logging.FileHandler("sm_ml_visualization.log", encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

class MLVisualizer:
    def __init__(self, save_dir="ml_visualizations"):
        """
        Инициализация визуализатора ML-результатов
        
        Args:
            save_dir: директория для сохранения визуализаций
        """
        self.save_dir = save_dir
        self.predictions_history = []  # Для хранения истории прогнозов
        self.max_history_size = 100    # Максимальный размер истории
        
        # Создаем директорию для визуализаций, если ее нет
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Настройка стиля графиков
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 7)
        plt.rcParams['font.size'] = 12
        
        logger.info("ML Visualizer инициализирован")
    
    def add_prediction_to_history(self, prediction_data):
        """
        Добавление прогноза в историю для последующей визуализации
        
        Args:
            prediction_data: словарь с данными о прогнозе {
                'timestamp': время прогноза,
                'pair': валютная пара,
                'direction': направление прогноза,
                'confidence': уверенность,
                'conventional_confidence': уверенность традиционного алгоритма,
                'ml_confidence': уверенность ML,
                'lstm_confidence': уверенность LSTM,
                'result': результат (True/False/None),
                'features': дополнительные признаки
            }
        """
        # Добавляем временную метку, если ее нет
        if 'timestamp' not in prediction_data:
            prediction_data['timestamp'] = datetime.now()
        
        # Добавляем в начало списка для хронологического порядка
        self.predictions_history.insert(0, prediction_data)
        
        # Ограничиваем размер истории
        if len(self.predictions_history) > self.max_history_size:
            self.predictions_history = self.predictions_history[:self.max_history_size]
    
    def save_predictions_history(self, filename="predictions_history.json"):
        """
        Сохранение истории прогнозов в JSON-файл
        
        Args:
            filename: имя файла для сохранения
        """
        try:
            # Преобразуем datetime объекты в строки
            serializable_history = []
            for pred in self.predictions_history:
                pred_copy = pred.copy()
                if isinstance(pred_copy.get('timestamp'), datetime):
                    pred_copy['timestamp'] = pred_copy['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                serializable_history.append(pred_copy)
            
            file_path = os.path.join(self.save_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, indent=2)
            
            logger.info(f"История прогнозов сохранена в {file_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении истории прогнозов: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_predictions_history(self, filename="predictions_history.json"):
        """
        Загрузка истории прогнозов из JSON-файла
        
        Args:
            filename: имя файла для загрузки
            
        Returns:
            bool: успешность загрузки
        """
        try:
            file_path = os.path.join(self.save_dir, filename)
            if not os.path.exists(file_path):
                logger.warning(f"Файл {file_path} не найден")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                serialized_history = json.load(f)
            
            # Преобразуем строки обратно в datetime
            history = []
            for pred in serialized_history:
                if 'timestamp' in pred and isinstance(pred['timestamp'], str):
                    try:
                        pred['timestamp'] = datetime.strptime(pred['timestamp'], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        pred['timestamp'] = datetime.now()
                history.append(pred)
            
            self.predictions_history = history
            logger.info(f"Загружено {len(history)} прогнозов из {file_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при загрузке истории прогнозов: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def plot_confidence_comparison(self, save=False, show=True):
        """
        Построение графика сравнения уверенности разных алгоритмов
        
        Args:
            save: сохранить график в файл
            show: показать график
            
        Returns:
            matplotlib.figure.Figure: объект графика
        """
        if not self.predictions_history:
            logger.warning("История прогнозов пуста")
            return None
        
        try:
            # Фильтруем только завершенные прогнозы
            complete_predictions = [p for p in self.predictions_history if p.get('result') is not None]
            
            if not complete_predictions:
                logger.warning("Нет завершенных прогнозов для визуализации")
                return None
            
            # Создаем DataFrame для удобства работы
            data = []
            for pred in complete_predictions:
                data.append({
                    'timestamp': pred.get('timestamp', datetime.now()),
                    'conventional': pred.get('conventional_confidence', 0),
                    'ml': pred.get('ml_confidence', 0),
                    'lstm': pred.get('lstm_confidence', 0),
                    'combined': pred.get('confidence', 0),
                    'correct': pred.get('result', False)
                })
            
            df = pd.DataFrame(data)
            
            # Сортируем по времени
            df = df.sort_values(by='timestamp')
            
            # Создаем график
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Отдельные линии для разных алгоритмов
            ax.plot(df['timestamp'], df['conventional'], 'b-', label='Традиционный')
            ax.plot(df['timestamp'], df['ml'], 'g-', label='ML')
            ax.plot(df['timestamp'], df['lstm'], 'r-', label='LSTM')
            ax.plot(df['timestamp'], df['combined'], 'k--', label='Объединенный')
            
            # Отмечаем правильные и неправильные прогнозы
            correct_df = df[df['correct'] == True]
            incorrect_df = df[df['correct'] == False]
            
            ax.scatter(correct_df['timestamp'], correct_df['combined'], 
                      color='green', s=100, marker='^', label='Правильно')
            ax.scatter(incorrect_df['timestamp'], incorrect_df['combined'], 
                      color='red', s=100, marker='x', label='Неправильно')
            
            # Настройки графика
            ax.set_title('Сравнение уверенности алгоритмов прогнозирования')
            ax.set_xlabel('Время')
            ax.set_ylabel('Уверенность')
            ax.legend()
            ax.grid(True)
            
            # Поворачиваем метки даты для лучшей читаемости
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Сохраняем график, если требуется
            if save:
                file_path = os.path.join(self.save_dir, 'confidence_comparison.png')
                plt.savefig(file_path)
                logger.info(f"График сохранен в {file_path}")
            
            # Показываем график, если требуется
            if show:
                plt.show()
            
            return fig
        except Exception as e:
            logger.error(f"Ошибка при построении графика точности алгоритмов: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def plot_market_session_performance(self, save=False, show=True):
        """
        Построение графика эффективности по торговым сессиям
        
        Args:
            save: сохранить график в файл
            show: показать график
            
        Returns:
            matplotlib.figure.Figure: объект графика
        """
        if not self.predictions_history:
            logger.warning("История прогнозов пуста")
            return None
        
        try:
            # Фильтруем только завершенные прогнозы
            complete_predictions = [p for p in self.predictions_history if p.get('result') is not None]
            
            if not complete_predictions:
                logger.warning("Нет завершенных прогнозов для визуализации")
                return None
            
            # Группируем по торговым сессиям
            sessions = ['asia', 'europe', 'us', 'overlap', 'closed']
            session_stats = {session: {'correct': 0, 'incorrect': 0} for session in sessions}
            
            for pred in complete_predictions:
                # Определяем сессию из признаков
                session = 'unknown'
                for s in sessions:
                    if pred.get(f'session_{s}', 0) == 1 or pred.get('market_session') == s:
                        session = s
                        break
                
                if session in session_stats:
                    if pred.get('result', False):
                        session_stats[session]['correct'] += 1
                    else:
                        session_stats[session]['incorrect'] += 1
            
            # Вычисляем точность для каждой сессии
            accuracies = []
            totals = []
            labels = []
            
            for session in sessions:
                correct = session_stats[session]['correct']
                incorrect = session_stats[session]['incorrect']
                total = correct + incorrect
                
                if total > 0:
                    accuracy = (correct / total) * 100
                    accuracies.append(accuracy)
                    totals.append(total)
                    labels.append(session.capitalize())
            
            # Создаем график
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Создаем столбчатую диаграмму
            bars = ax.bar(labels, accuracies, color='skyblue')
            
            # Добавляем метки со значениями
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            # Настройки графика
            ax.set_title('Точность прогнозов по торговым сессиям')
            ax.set_ylabel('Точность (%)')
            ax.set_ylim(0, 100)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Добавляем количество прогнозов для каждой сессии
            for i, (session, total) in enumerate(zip(labels, totals)):
                ax.annotate(f"n={total}", (i, 5), ha='center', va='bottom', color='black')
            
            # Сохраняем график, если требуется
            if save:
                file_path = os.path.join(self.save_dir, 'session_accuracy.png')
                plt.savefig(file_path)
                logger.info(f"График по сессиям сохранен в {file_path}")
            
            # Показываем график, если требуется
            if show:
                plt.show()
            
            return fig
        except Exception as e:
            logger.error(f"Ошибка при построении графика по сессиям: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def plot_noise_level_impact(self, save=False, show=True):
        """
        Построение графика влияния уровня шума на точность прогнозов
        
        Args:
            save: сохранить график в файл
            show: показать график
            
        Returns:
            matplotlib.figure.Figure: объект графика
        """
        if not self.predictions_history:
            logger.warning("История прогнозов пуста")
            return None
        
        try:
            # Фильтруем только завершенные прогнозы с информацией о шуме
            noise_predictions = [p for p in self.predictions_history 
                                if p.get('result') is not None and 'noise_level' in p]
            
            if not noise_predictions or len(noise_predictions) < 10:
                logger.warning("Недостаточно данных о шуме для визуализации")
                return None
            
            # Создаем DataFrame для удобства работы
            data = []
            for pred in noise_predictions:
                data.append({
                    'noise_level': pred.get('noise_level', 0),
                    'correct': pred.get('result', False)
                })
            
            df = pd.DataFrame(data)
            
            # Группируем по уровню шума с разбиением на 5 категорий
            df['noise_category'] = pd.qcut(df['noise_level'], 5, labels=False)
            
            # Вычисляем границы категорий для меток
            noise_bins = pd.qcut(df['noise_level'], 5, retbins=True)[1]
            noise_labels = [f"{noise_bins[i]:.3f}-{noise_bins[i+1]:.3f}" for i in range(len(noise_bins)-1)]
            
            # Вычисляем точность для каждой категории шума
            accuracy_by_noise = df.groupby('noise_category')['correct'].agg(['mean', 'count'])
            accuracy_by_noise['accuracy'] = accuracy_by_noise['mean'] * 100
            
            # Создаем график
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Создаем столбчатую диаграмму
            bars = ax.bar(noise_labels, accuracy_by_noise['accuracy'], color='lightgreen')
            
            # Добавляем метки со значениями
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            # Настройки графика
            ax.set_title('Влияние уровня шума на точность прогнозов')
            ax.set_xlabel('Уровень шума')
            ax.set_ylabel('Точность (%)')
            ax.set_ylim(0, 100)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            
            # Добавляем количество прогнозов для каждой категории
            for i, count in enumerate(accuracy_by_noise['count']):
                ax.annotate(f"n={count}", (i, 5), ha='center', va='bottom', color='black')
            
            # Добавляем тренд точности в зависимости от шума
            x = np.arange(len(noise_labels))
            z = np.polyfit(x, accuracy_by_noise['accuracy'], 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.8, label=f"Тренд: {z[0]:.2f}x + {z[1]:.2f}")
            ax.legend()
            
            # Сохраняем график, если требуется
            if save:
                file_path = os.path.join(self.save_dir, 'noise_impact.png')
                plt.savefig(file_path)
                logger.info(f"График влияния шума сохранен в {file_path}")
            
            # Показываем график, если требуется
            if show:
                plt.show()
            
            return fig
        except Exception as e:
            logger.error(f"Ошибка при построении графика влияния шума: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def generate_comprehensive_report(self, save_path=None, show=False):
        """
        Генерация комплексного отчета о производительности различных алгоритмов
        
        Args:
            save_path: путь для сохранения отчета (по умолчанию: ml_report.pdf)
            show: показывать графики в процессе генерации
            
        Returns:
            str: путь к сохраненному отчету или None в случае ошибки
        """
        try:
            if not self.predictions_history:
                logger.warning("История прогнозов пуста")
                return None
            
            if not save_path:
                save_path = os.path.join(self.save_dir, 'ml_report.pdf')
            
            # Создаем временную директорию для графиков
            temp_dir = os.path.join(self.save_dir, 'temp_report')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Генерируем все графики и сохраняем их
            print("Генерация графиков для отчета...")
            
            # 1. График сравнения уверенности
            confidence_fig = self.plot_confidence_comparison(save=True, show=show)
            confidence_path = os.path.join(temp_dir, 'confidence_comparison.png')
            if confidence_fig:
                confidence_fig.savefig(confidence_path)
                plt.close(confidence_fig)
            
            # 2. График точности алгоритмов
            accuracy_fig = self.plot_algorithm_accuracy(save=True, show=show)
            accuracy_path = os.path.join(temp_dir, 'algorithm_accuracy.png')
            if accuracy_fig:
                accuracy_fig.savefig(accuracy_path)
                plt.close(accuracy_fig)
            
            # 3. График производительности по сессиям
            session_fig = self.plot_market_session_performance(save=True, show=show)
            session_path = os.path.join(temp_dir, 'session_performance.png')
            if session_fig:
                session_fig.savefig(session_path)
                plt.close(session_fig)
            
            # 4. График влияния шума
            noise_fig = self.plot_noise_level_impact(save=True, show=show)
            noise_path = os.path.join(temp_dir, 'noise_impact.png')
            if noise_fig:
                noise_fig.savefig(noise_path)
                plt.close(noise_fig)
            
            # Создаем PDF отчет
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
                from reportlab.lib.styles import getSampleStyleSheet
                
                doc = SimpleDocTemplate(save_path, pagesize=letter)
                styles = getSampleStyleSheet()
                
                # Подготовка контента отчета
                content = []
                
                # Заголовок
                title = Paragraph("Отчет о производительности ML-компонентов ScalpMaster", styles['Title'])
                content.append(title)
                content.append(Spacer(1, 12))
                
                # Дата и время отчета
                date_text = Paragraph(f"Отчет сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
                content.append(date_text)
                content.append(Spacer(1, 12))
                
                # Статистика записей
                stats_text = Paragraph(f"Всего проанализировано прогнозов: {len(self.predictions_history)}", styles['Normal'])
                content.append(stats_text)
                content.append(Spacer(1, 24))
                
                # Добавляем графики
                if os.path.exists(confidence_path):
                    content.append(Paragraph("Сравнение уверенности алгоритмов", styles['Heading2']))
                    content.append(Spacer(1, 12))
                    content.append(Image(confidence_path, width=450, height=300))
                    content.append(Spacer(1, 24))
                
                if os.path.exists(accuracy_path):
                    content.append(Paragraph("Точность алгоритмов прогнозирования", styles['Heading2']))
                    content.append(Spacer(1, 12))
                    content.append(Image(accuracy_path, width=450, height=300))
                    content.append(Spacer(1, 24))
                
                if os.path.exists(session_path):
                    content.append(Paragraph("Производительность по торговым сессиям", styles['Heading2']))
                    content.append(Spacer(1, 12))
                    content.append(Image(session_path, width=450, height=300))
                    content.append(Spacer(1, 24))
                
                if os.path.exists(noise_path):
                    content.append(Paragraph("Влияние уровня шума на точность прогнозов", styles['Heading2']))
                    content.append(Spacer(1, 12))
                    content.append(Image(noise_path, width=450, height=300))
                    content.append(Spacer(1, 24))
                
                # Сохраняем документ
                doc.build(content)
                
                logger.info(f"Отчет успешно сохранен в {save_path}")
                return save_path
                
            except ImportError:
                logger.warning("Не удалось импортировать reportlab. Отчет не создан.")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при генерации отчета: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        except Exception as e:
            logger.error(f"Ошибка при построении графика сравнения уверенности: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def plot_algorithm_accuracy(self, save=False, show=True):
        """
        Построение графика точности разных алгоритмов
        
        Args:
            save: сохранить график в файл
            show: показать график
            
        Returns:
            matplotlib.figure.Figure: объект графика
        """
        if not self.predictions_history:
            logger.warning("История прогнозов пуста")
            return None
        
        try:
            # Фильтруем только завершенные прогнозы
            complete_predictions = [p for p in self.predictions_history if p.get('result') is not None]
            
            if not complete_predictions:
                logger.warning("Нет завершенных прогнозов для визуализации")
                return None
            
            # Функция для определения правильности прогноза алгоритма
            def is_correct(pred, algorithm):
                # Получаем направление прогноза алгоритма (если есть)
                if f"{algorithm}_direction" in pred:
                    algo_direction = pred[f"{algorithm}_direction"]
                else:
                    # Если направление не указано, но есть уверенность, определяем направление косвенно
                    if algorithm == "conventional" and "conventional_confidence" in pred:
                        algo_direction = pred.get("direction", None)  # Предполагаем, что направление совпадает с общим
                    elif algorithm == "ml" and "ml_confidence" in pred and pred["ml_confidence"] > 0:
                        algo_direction = pred.get("ml_direction", pred.get("direction", None))
                    elif algorithm == "lstm" and "lstm_confidence" in pred and pred["lstm_confidence"] > 0:
                        algo_direction = pred.get("lstm_direction", pred.get("direction", None))
                    else:
                        return None  # Нет данных для этого алгоритма
                
                # Если направление не определено, не можем оценить правильность
                if algo_direction is None:
                    return None
                
                # Получаем фактическое направление цены
                actual_direction = "up" if pred.get("price_moved_up", False) else "down"
                
                # Сравниваем прогноз с фактом
                return algo_direction == actual_direction
            
            # Вычисляем точность для каждого алгоритма
            algorithms = ["conventional", "ml", "lstm", "combined"]
            accuracy_data = {}
            
            for algorithm in algorithms:
                correct = 0
                total = 0
                
                for pred in complete_predictions:
                    result = is_correct(pred, algorithm)
                    if result is not None:  # Учитываем только те прогнозы, где был этот алгоритм
                        total += 1
                        if result:
                            correct += 1
                
                accuracy = correct / total if total > 0 else 0
                accuracy_data[algorithm] = {"correct": correct, "total": total, "accuracy": accuracy}
            
            # Создаем график
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Данные для графика
            labels = [alg.capitalize() for alg in algorithms]
            accuracy_values = [accuracy_data[alg]["accuracy"] * 100 for alg in algorithms]
            
            # Цвета для каждого алгоритма
            colors = ['blue', 'green', 'red', 'purple']
            
            # Создаем столбчатую диаграмму
            bars = ax.bar(labels, accuracy_values, color=colors, alpha=0.7)
            
            # Добавляем метки со значениями
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            # Настройки графика
            ax.set_title('Точность алгоритмов прогнозирования')
            ax.set_ylabel('Точность (%)')
            ax.set_ylim(0, 100)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Добавляем аннотацию с количеством прогнозов
            for i, alg in enumerate(algorithms):
                count_text = f"{accuracy_data[alg]['correct']}/{accuracy_data[alg]['total']}"
                ax.annotate(count_text, (i, 5), ha='center', va='bottom', color='black')
            
            # Сохраняем график, если требуется
            if save:
                file_path = os.path.join(self.save_dir, 'algorithm_accuracy.png')
                plt.savefig(file_path)
                logger.info(f"График точности сохранен в {file_path}")
            
            # Показываем график, если требуется
            if show:
                plt.show()
            
            return fig
        except Exception as e:
            logger.error(f"Ошибка при построении графика точности алгоритмов: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def plot_prediction_trends(self, days=7, save=False, show=True):
        """
        Построение графика трендов прогнозов с течением времени
        
        Args:
            days: количество дней для анализа
            save: сохранить график в файл
            show: показать график
            
        Returns:
            matplotlib.figure.Figure: объект графика
        """
        if not self.predictions_history:
            logger.warning("История прогнозов пуста")
            return None
        
        try:
            # Фильтруем прогнозы за указанный период
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_predictions = [p for p in self.predictions_history 
                                 if p.get('timestamp', datetime.now()) >= cutoff_date]
            
            if not recent_predictions:
                logger.warning(f"Нет прогнозов за последние {days} дней")
                return None
            
            # Группируем по дням
            predictions_by_day = {}
            for pred in recent_predictions:
                timestamp = pred.get('timestamp', datetime.now())
                day_key = timestamp.strftime('%Y-%m-%d')
                
                if day_key not in predictions_by_day:
                    predictions_by_day[day_key] = []
                
                predictions_by_day[day_key].append(pred)
            
            # Вычисляем ежедневные показатели
            dates = []
            accuracy_values = []
            confidence_values = []
            count_values = []
            
            for day, preds in sorted(predictions_by_day.items()):
                # Фильтруем только завершенные прогнозы
                completed = [p for p in preds if p.get('result') is not None]
                
                if completed:
                    # Точность прогнозов
                    accuracy = sum(1 for p in completed if p.get('result', False)) / len(completed) * 100
                    
                    # Средняя уверенность
                    avg_confidence = sum(p.get('confidence', 0) for p in completed) / len(completed) * 100
                    
                    dates.append(day)
                    accuracy_values.append(accuracy)
                    confidence_values.append(avg_confidence)
                    count_values.append(len(completed))
            
            # Создаем график
            fig, ax1 = plt.subplots(figsize=(12, 7))
            
            # Основная ось для точности и уверенности
            line1 = ax1.plot(dates, accuracy_values, 'g-', marker='o', label='Точность (%)')
            line2 = ax1.plot(dates, confidence_values, 'b--', marker='s', label='Уверенность (%)')
            ax1.set_xlabel('Дата')
            ax1.set_ylabel('Процент (%)')
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='x', rotation=45)
            
            # Вторая ось для количества прогнозов
            ax2 = ax1.twinx()
            line3 = ax2.plot(dates, count_values, 'r:', marker='x', label='Количество прогнозов')
            ax2.set_ylabel('Количество прогнозов')
            ax2.set_ylim(0, max(count_values) * 1.2 if count_values else 10)
            
            # Добавляем легенду
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            
            # Настройки графика
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f'Тренды прогнозов за последние {days} дней')
            plt.tight_layout()
            
            # Сохраняем график, если требуется
            if save:
                file_path = os.path.join(self.save_dir, 'prediction_trends.png')
                plt.savefig(file_path)
                logger.info(f"График трендов прогнозов сохранен в {file_path}")
            
            # Показываем график, если требуется
            if show:
                plt.show()
            
            return fig
        except Exception as e:
            logger.error(f"Ошибка при построении графика трендов прогнозов: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def plot_feature_importance(self, algorithm='lstm', save=False, show=True):
        """
        Построение графика важности признаков для выбранного алгоритма
        
        Args:
            algorithm: алгоритм ('lstm', 'ml' или 'combined')
            save: сохранить график в файл
            show: показать график
            
        Returns:
            matplotlib.figure.Figure: объект графика
        """
        if not self.predictions_history:
            logger.warning("История прогнозов пуста")
            return None
        
        try:
            # Собираем все признаки из истории прогнозов
            all_features = {}
            for pred in self.predictions_history:
                # Для каждого прогноза собираем информацию о признаках и их влиянии
                if 'features' in pred:
                    features = pred.get('features', {})
                    for key, value in features.items():
                        if key not in all_features and not isinstance(value, dict):
                            all_features[key] = []
                        
                        if not isinstance(value, dict):
                            all_features[key].append(value)
            
            # Проверяем наличие достаточного количества данных
            if not all_features:
                logger.warning("Недостаточно данных о признаках для анализа")
                return None
            
            # Вычисляем корреляцию признаков с результатами прогнозов
            feature_correlation = {}
            for feature, values in all_features.items():
                if len(values) < 10:  # Игнорируем признаки с малым количеством данных
                    continue
                
                # Создаем DataFrame для удобства анализа
                data = []
                for i, pred in enumerate(self.predictions_history):
                    if i < len(values) and 'result' in pred and pred['result'] is not None:
                        data.append({
                            'feature': values[i],
                            'result': 1 if pred['result'] else 0,
                            'confidence': pred.get(f'{algorithm}_confidence', 0) 
                                          if algorithm != 'combined' else pred.get('confidence', 0)
                        })
                
                if data:
                    df = pd.DataFrame(data)
                    
                    # Корреляция с результатом
                    result_corr = df['feature'].corr(df['result'])
                    
                    # Корреляция с уверенностью
                    confidence_corr = df['feature'].corr(df['confidence'])
                    
                    # Сохраняем для отображения
                    feature_correlation[feature] = (result_corr, confidence_corr)
            
            # Сортируем признаки по корреляции с результатами
            sorted_features = sorted(feature_correlation.items(), 
                                    key=lambda x: abs(x[1][0]), reverse=True)
            
            # Ограничиваем количество признаков для отображения
            top_features = sorted_features[:15] if len(sorted_features) > 15 else sorted_features
            
            # Создаем график
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Данные для построения
            feature_names = [f[0] fo