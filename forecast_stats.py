"""
Модуль для визуализации статистики прогнозов в системе ScalpMaster (SM)
"""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import logging

# Настройка логирования
logger = logging.getLogger("forecast_stats")

def load_forecast_data(filename="forecast_log.csv", extended_filename="pattern_forecast_log.csv"):
    """
    Загрузка данных о прогнозах из CSV-файла с поддержкой нового расширенного формата
    
    Args:
        filename: путь к основному файлу с логами прогнозов
        extended_filename: путь к расширенному файлу с логами прогнозов
        
    Returns:
        tuple: (список прогнозов, словарь статистики) или (None, None) при ошибке
    """
    forecasts = []
    stats = {
        "Двойное дно": {"correct": 0, "incorrect": 0},
        "Двойная вершина": {"correct": 0, "incorrect": 0},
        "Восходящий клин": {"correct": 0, "incorrect": 0},
        "Нисходящий клин": {"correct": 0, "incorrect": 0},
        "Линия тренда": {"correct": 0, "incorrect": 0},
        "all": {"correct": 0, "incorrect": 0}
    }
    
    # Проверка основного файла логов
    if os.path.exists(filename):
        try:
            with open(filename, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Пропускаем заголовок
                
                for row in reader:
                    # Проверка формата данных
                    if len(row) < 8:
                        continue
                        
                    # Проверка на пустые значения результатов
                    if not row[6] or not row[7]:
                        continue
                    
                    try:
                        forecast = {
                            "time": datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'),
                            "pattern": row[1],
                            "confidence": float(row[2]),
                            "direction": row[3],
                            "message": row[4],
                            "check_time": datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S'),
                            "result": row[6],
                            "is_correct": row[7].lower() == 'true'
                        }
                        
                        forecasts.append(forecast)
                        
                        # Обновляем статистику
                        pattern = forecast["pattern"]
                        if pattern not in stats:
                            # Если встретился новый паттерн, добавляем его в статистику
                            stats[pattern] = {"correct": 0, "incorrect": 0}
                        
                        if forecast["is_correct"]:
                            stats[pattern]["correct"] += 1
                            stats["all"]["correct"] += 1
                        else:
                            stats[pattern]["incorrect"] += 1
                            stats["all"]["incorrect"] += 1
                    except ValueError:
                        pass
            
        except Exception:
            pass
    
    # Проверка расширенного файла логов (если основной файл пуст или отсутствует)
    if (not forecasts or len(forecasts) == 0) and os.path.exists(extended_filename):
        try:
            forecasts, stats = load_pattern_forecast_data(extended_filename)
        except Exception:
            pass
    
    # Если ни в одном из файлов нет данных
    if not forecasts:
        return None, None
    
    return forecasts, stats

def load_binary_stats(filename="binary_signals_log.csv"):
    """
    Загрузка данных о сигналах бинарных опционов
    
    Args:
        filename: путь к файлу с логами сигналов
        
    Returns:
        tuple: (список сигналов, словарь статистики) или (None, None) при ошибке
    """
    if not os.path.exists(filename):
        logger.warning(f"Файл {filename} не найден")
        return None, None
        
    signals = []
    stats = {
        "UP": {"win": 0, "loss": 0},
        "DOWN": {"win": 0, "loss": 0},
        "all": {"win": 0, "loss": 0}
    }
    
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Пропускаем заголовок
            
            logger.debug(f"Загрузка данных из файла: {filename}")
            logger.debug(f"Заголовок: {header}")
            
            for row in reader:
                # Проверка формата данных
                if len(row) < 8:
                    logger.warning(f"Строка имеет недостаточно столбцов: {row}")
                    continue
                    
                # Если результат не заполнен, пропускаем
                if not row[7]:
                    continue
                
                signal = {
                    "time": row[0],
                    "pair": row[1],
                    "timeframe": row[2],
                    "expiration": row[3],
                    "direction": row[4],
                    "confidence": float(row[5]) if row[5] else 0,
                    "sources": row[6],
                    "result": row[7],
                    "pnl": float(row[8]) if len(row) > 8 and row[8] else 0
                }
                
                signals.append(signal)
                
                # Обновляем статистику
                direction = signal["direction"]
                result = signal["result"]
                
                if direction not in stats:
                    stats[direction] = {"win": 0, "loss": 0}
                
                if result == "WIN":
                    stats[direction]["win"] += 1
                    stats["all"]["win"] += 1
                elif result == "LOSS":
                    stats[direction]["loss"] += 1
                    stats["all"]["loss"] += 1
            
            logger.info(f"Загружено сигналов: {len(signals)}")
            if logger.level <= logging.INFO:
                for direction in stats:
                    wins = stats[direction]["win"]
                    losses = stats[direction]["loss"]
                    total = wins + losses
                    if total > 0:
                        logger.info(f"{direction}: {wins}/{total} ({wins/total*100:.1f}%)")
        
        return signals, stats
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def show_forecast_stats(stats=None, filename="forecast_log.csv", extended_filename="pattern_forecast_log.csv"):
    """
    Отображение статистики прогнозов с поддержкой обоих форматов лог-файлов
    """
    if stats is None:
        # Пробуем загрузить данные из любого доступного файла
        try:
            data = load_forecast_data(filename)
            if data and data[0]:
                forecasts, stats = data
            else:
                # Если основной файл пустой, используем специальную функцию для расширенного файла
                forecasts, stats = load_pattern_forecast_data(extended_filename)
        except Exception as e:
            forecasts, stats = load_pattern_forecast_data(extended_filename)
            
        if not forecasts:
            print("Нет данных для отображения статистики")
            return
    
    # Проверка, есть ли данные для отображения
    any_data = False
    for pattern in stats:
        if stats[pattern]["correct"] > 0 or stats[pattern]["incorrect"] > 0:
            any_data = True
            break
    
    if not any_data:
        print("Нет данных для отображения статистики")
        return
    
    print("\nСтатистика прогнозов:")
    print("-" * 60)
    print(f"{'Паттерн':<25} {'Верных':<10} {'Неверных':<10} {'Точность':<10}")
    print("-" * 60)
    
    # Сначала показываем статистику по всем паттернам
    correct_all = stats["all"]["correct"]
    incorrect_all = stats["all"]["incorrect"]
    total_all = correct_all + incorrect_all
    accuracy_all = (correct_all / total_all) * 100 if total_all > 0 else 0
    
    print(f"{'Все паттерны':<25} {correct_all:<10} {incorrect_all:<10} {accuracy_all:.1f}%")
    print("-" * 60)
    
    # Затем показываем статистику по каждому паттерну отдельно (кроме 'all')
    for pattern in sorted(stats.keys()):
        if pattern == "all":
            continue
            
        correct = stats[pattern]["correct"]
        incorrect = stats[pattern]["incorrect"]
        total = correct + incorrect
        
        # Пропускаем паттерны без статистики
        if total == 0:
            continue
            
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        print(f"{pattern:<25} {correct:<10} {incorrect:<10} {accuracy:.1f}%")
    
    print("-" * 60)

def show_binary_stats(stats=None, filename="binary_signals_log.csv"):
    """
    Отображение статистики бинарных опционов
    
    Args:
        stats: словарь со статистикой или None для автоматической загрузки
        filename: путь к файлу с логами сигналов
    """
    if stats is None:
        data = load_binary_stats(filename)
        if data is None or data[0] is None:
            print("Нет данных для отображения статистики")
            return
        signals, stats = data
    
    # Проверка, есть ли данные для отображения
    any_data = False
    for direction in stats:
        if stats[direction]["win"] > 0 or stats[direction]["loss"] > 0:
            any_data = True
            break
    
    if not any_data:
        print("Нет данных для отображения статистики")
        return
    
    print("\nСтатистика бинарных опционов:")
    print("-" * 60)
    print(f"{'Направление':<15} {'Побед':<10} {'Поражений':<10} {'Процент побед':<15} {'P&L':<10}")
    print("-" * 60)
    
    total_pnl = 0
    
    for direction in stats:
        wins = stats[direction]["win"]
        losses = stats[direction]["loss"]
        total = wins + losses
        win_rate = (wins / total) * 100 if total > 0 else 0
        
        # Расчет P&L для направления можно добавить, если доступны данные
        
        print(f"{direction:<15} {wins:<10} {losses:<10} {win_rate:.1f}%{'':<15}")
    
    print("-" * 60)

def create_forecast_charts(stats=None, filename="forecast_log.csv", save_path="forecast_stats.png"):
    """
    Создание и сохранение графиков статистики прогнозов с поддержкой нового формата
    
    Args:
        stats: словарь со статистикой или None для автоматической загрузки
        filename: путь к файлу с логами прогнозов (или расширенному)
        save_path: путь для сохранения графика
        
    Returns:
        matplotlib.pyplot: объект графика или None при ошибке
    """
    if stats is None:
        # Пробуем загрузить данные из любого доступного файла
        try:
            base_data = load_forecast_data(filename)
            if base_data and base_data[0]:
                forecasts, stats = base_data
            else:
                # Используем специальную функцию для расширенного файла
                forecasts, stats = load_pattern_forecast_data("pattern_forecast_log.csv")
        except Exception:
            try:
                forecasts, stats = load_pattern_forecast_data("pattern_forecast_log.csv")
            except:
                return None
    
    # Проверка, есть ли данные для графиков
    total_forecasts = 0
    for pattern in stats:
        if pattern != "all":
            total_forecasts += stats[pattern]["correct"] + stats[pattern]["incorrect"]
    
    if total_forecasts == 0:
        return None
    
    # Отфильтруем "all" из списка паттернов и паттерны без данных
    patterns = []
    correct = []
    incorrect = []
    
    for p in stats.keys():
        if p != "all" and (stats[p]["correct"] > 0 or stats[p]["incorrect"] > 0):
            patterns.append(p)
            correct.append(stats[p]["correct"])
            incorrect.append(stats[p]["incorrect"])
    
    # Расчет общего количества и процента успешности
    totals = [c + i for c, i in zip(correct, incorrect)]
    percentages = [(c / t) * 100 if t > 0 else 0 for c, t in zip(correct, totals)]
    
    try:
        # Создание графика
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # График количества прогнозов
        x = np.arange(len(patterns))
        width = 0.35
        
        ax1.bar(x - width/2, correct, width, label='Верно')
        ax1.bar(x + width/2, incorrect, width, label='Неверно')
        
        ax1.set_title('Количество прогнозов по паттернам')
        ax1.set_xticks(x)
        ax1.set_xticklabels(patterns, rotation=45, ha='right')
        ax1.legend()
        
        # График процента успешности
        ax2.bar(np.arange(len(patterns)), percentages, color='g')
        ax2.set_title('Точность прогнозов (%)')
        ax2.set_ylim([0, 100])
        ax2.set_xticks(np.arange(len(patterns)))
        ax2.set_xticklabels(patterns, rotation=45, ha='right')
        
        for i, p in enumerate(percentages):
            ax2.text(i, p + 2, f"{p:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(save_path)
        
        # Отображаем суммарную статистику
        total_correct = stats["all"]["correct"]
        total_incorrect = stats["all"]["incorrect"]
        total = total_correct + total_incorrect
        total_accuracy = (total_correct / total) * 100 if total > 0 else 0
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie([total_correct, total_incorrect], 
               labels=[f'Верно ({total_correct})', f'Неверно ({total_incorrect})'],
               colors=['g', 'r'],
               autopct='%1.1f%%',
               startangle=90)
        ax.set_title(f'Общая точность прогнозов: {total_accuracy:.1f}%')
        plt.savefig('total_accuracy.png')
        
        return plt
    except Exception:
        return None

def create_trend_chart(filename="forecast_log.csv", extended_filename="pattern_forecast_log.csv", save_path="forecast_trend.png"):
    """
    Создание графика тренда точности прогнозов со временем
    
    Args:
        filename: путь к файлу с логами прогнозов
        extended_filename: путь к расширенному файлу с логами прогнозов
        save_path: путь для сохранения графика
        
    Returns:
        matplotlib.pyplot: объект графика или None при ошибке
    """
    data = load_forecast_data(filename, extended_filename)
    if data is None or data[0] is None:
        print("Нет данных для отображения тренда")
        return None
    forecasts, _ = data
    
    # Проверка, есть ли данные для графика
    if not forecasts:
        print("Недостаточно данных для отображения тренда")
        return None
    
    try:
        # Сортировка прогнозов по времени
        forecasts.sort(key=lambda x: x["time"])
        
        # Подготовка данных для графика
        dates = [f["time"] for f in forecasts]
        correct_cumulative = []
        total_cumulative = []
        accuracy_trend = []
        
        correct_count = 0
        total_count = 0
        
        for f in forecasts:
            total_count += 1
            if f["is_correct"]:
                correct_count += 1
            
            correct_cumulative.append(correct_count)
            total_cumulative.append(total_count)
            accuracy_trend.append((correct_count / total_count) * 100)
        
        # Создание графика
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # График накопительной статистики
        ax1.plot(dates, correct_cumulative, 'g-', label='Верных прогнозов')
        ax1.plot(dates, total_cumulative, 'b-', label='Всего прогнозов')
        ax1.set_title('Накопительная статистика прогнозов')
        ax1.legend()
        ax1.grid(True)
        
        # График тренда точности
        ax2.plot(dates, accuracy_trend, 'r-')
        ax2.set_title('Тренд точности прогнозов (%)')
        ax2.set_ylim([0, 100])
        ax2.grid(True)
        
        # Форматирование оси X для лучшего отображения дат
        fig.autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(save_path)
        
        return plt
    except Exception as e:
        logger.error(f"Ошибка при создании графика тренда: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
        
def load_pattern_forecast_data(filename="pattern_forecast_log.csv"):
    """
    Функция специально для чтения расширенного файла логов pattern_forecast_log.csv
    без вывода отладочной информации
    
    Args:
        filename: путь к файлу с логами
        
    Returns:
        tuple: (список прогнозов, словарь статистики)
    """
    forecasts = []
    stats = {
        "Двойное дно": {"correct": 0, "incorrect": 0},
        "Двойная вершина": {"correct": 0, "incorrect": 0},
        "Восходящий клин": {"correct": 0, "incorrect": 0},
        "Нисходящий клин": {"correct": 0, "incorrect": 0},
        "Линия тренда": {"correct": 0, "incorrect": 0},
        "Комплексный сигнал": {"correct": 0, "incorrect": 0},
        "all": {"correct": 0, "incorrect": 0}
    }
    
    if not os.path.exists(filename):
        return forecasts, stats
    
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Пропускаем заголовок
            
            # Определяем индексы столбцов, которые нам нужны
            indices = {}
            for i, col in enumerate(header):
                indices[col] = i
            
            for row in reader:
                # Проверяем, есть ли основные поля
                if len(row) < 5:
                    continue
                    
                try:
                    # Определяем, какой тип записи это - с оценкой результата или без
                    has_result = False
                    result_value = None
                    is_correct = None
                    
                    # Пытаемся найти столбцы "Результат" и "Правильно"
                    if "Результат" in indices and indices["Результат"] < len(row) and row[indices["Результат"]]:
                        has_result = True
                        result_value = row[indices["Результат"]]
                        
                    if "Правильно" in indices and indices["Правильно"] < len(row) and row[indices["Правильно"]]:
                        is_correct = row[indices["Правильно"]].lower() == 'true'
                    
                    # Также проверяем, есть ли "Верно"/"Неверно" в строке
                    for i, val in enumerate(row):
                        if val == "Верно":
                            has_result = True
                            result_value = "Верно" 
                            is_correct = True
                        elif val == "Неверно":
                            has_result = True
                            result_value = "Неверно"
                            is_correct = False
                    
                    # Если нет результата, пропускаем строку
                    if not has_result or is_correct is None:
                        continue
                    
                    # Определяем паттерн
                    pattern = row[1] if len(row) > 1 else "Неизвестный"
                    
                    # Добавляем запись о прогнозе
                    forecast = {
                        "time": datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S') if row[0] else datetime.now(),
                        "pattern": pattern,
                        "confidence": float(row[2]) if len(row) > 2 and row[2] else 0.0,
                        "direction": row[3] if len(row) > 3 else "",
                        "message": "",
                        "check_time": datetime.now(),
                        "result": result_value,
                        "is_correct": is_correct
                    }
                    
                    forecasts.append(forecast)
                    
                    # Обновляем статистику
                    if pattern.startswith("Линия тренда"):
                        pattern_key = "Линия тренда"
                    elif pattern not in stats:
                        # Если новый паттерн, добавляем его в статистику
                        stats[pattern] = {"correct": 0, "incorrect": 0}
                        pattern_key = pattern
                    else:
                        pattern_key = pattern
                    
                    if is_correct:
                        stats[pattern_key]["correct"] += 1
                        stats["all"]["correct"] += 1
                    else:
                        stats[pattern_key]["incorrect"] += 1
                        stats["all"]["incorrect"] += 1
                        
                except Exception:
                    pass
    
    except Exception:
        pass
    
    # Удаляем паттерны без данных
    for pattern in list(stats.keys()):
        if pattern != "all" and stats[pattern]["correct"] + stats[pattern]["incorrect"] == 0:
            del stats[pattern]
    
    return forecasts, stats

def show_forecast_stats(stats=None, filename="forecast_log.csv", extended_filename="pattern_forecast_log.csv"):
    """
    Отображение статистики прогнозов с поддержкой обоих форматов лог-файлов
    
    Args:
        stats: словарь со статистикой или None для автоматической загрузки
        filename: путь к основному файлу с логами прогнозов
        extended_filename: путь к расширенному файлу с логами прогнозов
    """
    if stats is None:
        # Сначала пробуем загрузить данные из основного файла
        try:
            data = load_forecast_data(filename)
            if data and data[0]:
                forecasts, stats = data
            else:
                # Если основной файл пустой, используем специальную функцию для расширенного файла
                forecasts, stats = load_pattern_forecast_data(extended_filename)
        except Exception:
            forecasts, stats = load_pattern_forecast_data(extended_filename)
            
        if not forecasts:
            print("Нет данных для отображения статистики")
            return
    
    # Проверка, есть ли данные для отображения
    any_data = False
    for pattern in stats:
        if stats[pattern]["correct"] > 0 or stats[pattern]["incorrect"] > 0:
            any_data = True
            break
    
    if not any_data:
        print("Нет данных для отображения статистики")
        return
    
    print("\nСтатистика прогнозов:")
    print("-" * 60)
    print(f"{'Паттерн':<25} {'Верных':<10} {'Неверных':<10} {'Точность':<10}")
    print("-" * 60)
    
    # Сначала показываем статистику по всем паттернам
    correct_all = stats["all"]["correct"]
    incorrect_all = stats["all"]["incorrect"]
    total_all = correct_all + incorrect_all
    accuracy_all = (correct_all / total_all) * 100 if total_all > 0 else 0
    
    print(f"{'Все паттерны':<25} {correct_all:<10} {incorrect_all:<10} {accuracy_all:.1f}%")
    print("-" * 60)
    
    # Затем показываем статистику по каждому паттерну отдельно (кроме 'all')
    for pattern in sorted(stats.keys()):
        if pattern == "all":
            continue
            
        correct = stats[pattern]["correct"]
        incorrect = stats[pattern]["incorrect"]
        total = correct + incorrect
        
        # Пропускаем паттерны без статистики
        if total == 0:
            continue
            
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        print(f"{pattern:<25} {correct:<10} {incorrect:<10} {accuracy:.1f}%")
    
    print("-" * 60)