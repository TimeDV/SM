"""
Модуль для расширенного логирования паттернов и прогнозов в системе ScalpMaster (SM)
"""

import csv
import os
import logging
from datetime import datetime
import json

# Настройка логирования
logger = logging.getLogger("pattern_logger")

class PatternLogger:
    def __init__(self, log_file="pattern_forecast_log.csv"):
        """
        Инициализация логгера паттернов и прогнозов
        
        Args:
            log_file: путь к файлу для записи логов
        """
        self.log_file = log_file
        self.init_log_file()
    
    def init_log_file(self):
        """Инициализация файла логирования с расширенными полями"""
        try:
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Расширенные заголовки для детального логирования
                    writer.writerow([
                        'Время прогноза', 
                        'Паттерн', 
                        'Уверенность', 
                        'Направление', 
                        'Предсказание', 
                        'Валютная пара',
                        'Таймфрейм',
                        'Время экспирации',
                        'Ключевые точки',  # JSON с координатами ключевых точек
                        'Доп. индикаторы', # Дополнительные индикаторы (RSI, Bollinger и т.д.)
                        'Время проверки', 
                        'Результат', 
                        'Правильно',
                        'Начальная цена',
                        'Конечная цена',
                        'Процент изменения'
                    ])
                logger.info(f"Создан новый лог-файл паттернов и прогнозов: {self.log_file}")
        except Exception as e:
            logger.error(f"Ошибка при инициализации лог-файла: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def log_pattern_forecast(self, forecast_data):
        """
        Запись подробной информации о прогнозе в CSV-файл
        
        Args:
            forecast_data: словарь с данными о прогнозе, должен содержать:
                time: время прогноза (datetime)
                pattern: название паттерна
                confidence: уверенность (0.0-1.0)
                direction: направление прогноза
                message: текст предсказания
                currency_pair: валютная пара (может быть None)
                timeframe: таймфрейм в минутах (может быть None)
                expiration: время экспирации в минутах (может быть None)
                key_points: словарь с ключевыми точками (может быть None)
                indicators: словарь с показаниями индикаторов (может быть None)
                check_time: время проверки прогноза (может быть None)
                result: результат (может быть None)
                is_correct: верность прогноза (может быть None)
                initial_price: начальная цена (может быть None)
                final_price: конечная цена (может быть None)
        """
        try:
            # Подготовка данных для записи
            key_points_json = json.dumps(forecast_data.get('key_points', {})) if forecast_data.get('key_points') else ''
            indicators_json = json.dumps(forecast_data.get('indicators', {})) if forecast_data.get('indicators') else ''
            
            # Расчет процента изменения
            percent_change = ''
            if forecast_data.get('initial_price') is not None and forecast_data.get('final_price') is not None:
                initial = forecast_data['initial_price']
                final = forecast_data['final_price']
                if initial != 0:
                    # Инвертируем, так как на экране Y растет вниз
                    percent_change = ((initial - final) / initial) * 100
            
            # Форматирование времени
            forecast_time = forecast_data.get('time', datetime.now())
            if isinstance(forecast_time, datetime):
                forecast_time_str = forecast_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                forecast_time_str = str(forecast_time)
            
            check_time = forecast_data.get('check_time')
            if check_time:
                if isinstance(check_time, datetime):
                    check_time_str = check_time.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    check_time_str = str(check_time)
            else:
                check_time_str = ''
            
            # Подготовка строки для записи
            row = [
                forecast_time_str,
                forecast_data.get('pattern', ''),
                f"{forecast_data.get('confidence', 0):.2f}",
                forecast_data.get('direction', ''),
                forecast_data.get('message', ''),
                forecast_data.get('currency_pair', ''),
                forecast_data.get('timeframe', ''),
                forecast_data.get('expiration', ''),
                key_points_json,
                indicators_json,
                check_time_str,
                forecast_data.get('result', ''),
                str(forecast_data.get('is_correct', '')),
                forecast_data.get('initial_price', ''),
                forecast_data.get('final_price', ''),
                f"{percent_change:.2f}" if percent_change != '' else ''
            ]
            
            # Запись в файл
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            logger.info(f"Запись о прогнозе успешно добавлена: {forecast_data.get('pattern')} - {forecast_data.get('direction')}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при записи в лог-файл: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def update_forecast_result(self, forecast_time, result_data):
        """
        Обновление информации о результате прогноза
        
        Args:
            forecast_time: время прогноза (строка в формате '%Y-%m-%d %H:%M:%S' или datetime)
            result_data: словарь с результатами:
                result: результат (текст)
                is_correct: правильность прогноза (bool)
                final_price: конечная цена
                check_time: время проверки (опционально)
        """
        try:
            if isinstance(forecast_time, datetime):
                forecast_time = forecast_time.strftime('%Y-%m-%d %H:%M:%S')
                
            # Временный файл для записи обновленных данных
            temp_file = f"{self.log_file}.temp"
            updated = False
            
            with open(self.log_file, 'r', newline='', encoding='utf-8') as input_file, \
                 open(temp_file, 'w', newline='', encoding='utf-8') as output_file:
                
                reader = csv.reader(input_file)
                writer = csv.writer(output_file)
                
                # Записываем заголовок
                header = next(reader)
                writer.writerow(header)
                
                # Индексы нужных колонок
                time_idx = 0  # Время прогноза
                check_time_idx = 10  # Время проверки
                result_idx = 11  # Результат
                correct_idx = 12  # Правильно
                final_price_idx = 14  # Конечная цена
                percent_change_idx = 15  # Процент изменения
                
                # Обновляем нужную строку
                for row in reader:
                    if row[time_idx] == forecast_time:
                        # Обновляем результат
                        row[result_idx] = result_data.get('result', row[result_idx])
                        row[correct_idx] = str(result_data.get('is_correct', row[correct_idx]))
                        
                        # Обновляем время проверки, если предоставлено
                        if 'check_time' in result_data:
                            check_time = result_data['check_time']
                            if isinstance(check_time, datetime):
                                row[check_time_idx] = check_time.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                row[check_time_idx] = str(check_time)
                        
                        # Обновляем конечную цену, если предоставлена
                        if 'final_price' in result_data:
                            row[final_price_idx] = str(result_data['final_price'])
                            
                            # Пересчитываем процент изменения, если есть начальная цена
                            if row[13] and row[final_price_idx]:  # Проверяем, что начальная и конечная цены не пустые
                                try:
                                    initial_price = float(row[13])
                                    final_price = float(row[final_price_idx])
                                    if initial_price != 0:
                                        # Инвертируем, так как на экране Y растет вниз
                                        percent_change = ((initial_price - final_price) / initial_price) * 100
                                        row[percent_change_idx] = f"{percent_change:.2f}"
                                except ValueError:
                                    pass
                        
                        updated = True
                    
                    writer.writerow(row)
            
            # Заменяем исходный файл на обновленный только если обновление произошло
            if updated:
                os.replace(temp_file, self.log_file)
                logger.info(f"Результат прогноза от {forecast_time} обновлен")
            else:
                os.remove(temp_file)
                logger.warning(f"Прогноз с временем {forecast_time} не найден в логе")
                
            return updated
        except Exception as e:
            logger.error(f"Ошибка при обновлении результата: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_pattern_statistics(self):
        """
        Получение статистики по паттернам из лог-файла
        
        Returns:
            dict: словарь со статистикой по паттернам
        """
        try:
            if not os.path.exists(self.log_file):
                logger.warning(f"Файл логов {self.log_file} не найден")
                return {}
            
            stats = {}
            with open(self.log_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Пропускаем заголовок
                
                # Считываем данные и обновляем статистику
                for row in reader:
                    if len(row) < 13:  # Проверяем, что достаточно столбцов
                        continue
                    
                    pattern = row[1]  # Название паттерна
                    is_correct = row[12].lower() == 'true'  # Правильность прогноза
                    
                    # Обновляем статистику по паттерну
                    if pattern not in stats:
                        stats[pattern] = {"correct": 0, "incorrect": 0, "total": 0}
                    
                    stats[pattern]["total"] += 1
                    if is_correct:
                        stats[pattern]["correct"] += 1
                    else:
                        stats[pattern]["incorrect"] += 1
            
            # Добавляем общую статистику по всем паттернам
            total_correct = sum(stats[pattern]["correct"] for pattern in stats)
            total_incorrect = sum(stats[pattern]["incorrect"] for pattern in stats)
            total = total_correct + total_incorrect
            
            stats["all"] = {
                "correct": total_correct,
                "incorrect": total_incorrect,
                "total": total
            }
            
            # Добавляем процент успешности
            for pattern in stats:
                total_pattern = stats[pattern]["total"]
                if total_pattern > 0:
                    stats[pattern]["success_rate"] = (stats[pattern]["correct"] / total_pattern) * 100
                else:
                    stats[pattern]["success_rate"] = 0
            
            return stats
        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def get_currency_pair_statistics(self):
        """
        Получение статистики по валютным парам из лог-файла
        
        Returns:
            dict: словарь со статистикой по валютным парам
        """
        try:
            if not os.path.exists(self.log_file):
                logger.warning(f"Файл логов {self.log_file} не найден")
                return {}
            
            stats = {}
            with open(self.log_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Пропускаем заголовок
                
                # Считываем данные и обновляем статистику
                for row in reader:
                    if len(row) < 13:  # Проверяем, что достаточно столбцов
                        continue
                    
                    currency_pair = row[5]  # Валютная пара
                    if not currency_pair:
                        currency_pair = "Unknown"
                        
                    is_correct = row[12].lower() == 'true'  # Правильность прогноза
                    
                    # Обновляем статистику по валютной паре
                    if currency_pair not in stats:
                        stats[currency_pair] = {"correct": 0, "incorrect": 0, "total": 0}
                    
                    stats[currency_pair]["total"] += 1
                    if is_correct:
                        stats[currency_pair]["correct"] += 1
                    else:
                        stats[currency_pair]["incorrect"] += 1
            
            # Добавляем процент успешности
            for pair in stats:
                total_pair = stats[pair]["total"]
                if total_pair > 0:
                    stats[pair]["success_rate"] = (stats[pair]["correct"] / total_pair) * 100
                else:
                    stats[pair]["success_rate"] = 0
            
            return stats
        except Exception as e:
            logger.error(f"Ошибка при получении статистики по валютным парам: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
