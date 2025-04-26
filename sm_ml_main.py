"""
ScalpMaster (SM) с интеграцией искусственного интеллекта.
Главный файл с расширенной функциональностью ML.
"""

import os
import sys
import csv
import matplotlib.pyplot as plt
import logging
import time
from datetime import datetime

# Импорт базовых модулей ScalpMaster
from binary_analyzer import BinaryPatternAnalyzer
from currency_profiles import CURRENCY_PROFILES
import forecast_stats
from visualization import display_binary_analysis
from screen_capture import select_monitor_area

# Импорт ML-модулей
from ml_integration import MLIntegration
from ml_pattern_analyzer import MLPatternAnalyzer 
from ml_data_processor import MLDataProcessor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler("sm_ml_app.log", encoding='utf-8')
    ]
)

# Отключаем вывод в консоль логов с уровнем INFO и ниже от всех модулей
for logger_name in ['screen_capture', 'candle_detection', 'pattern_detection', 
                    'visualization', 'binary_analyzer', 'simple_analyzer', 
                    'forecast_stats', 'ml_integration', 'ml_pattern_analyzer', 
                    'ml_data_processor']:
    module_logger = logging.getLogger(logger_name)
    module_logger.propagate = False  # Отключаем передачу логов в корневой логгер
    
    # Добавляем файловый обработчик для сохранения всех логов в файл
    file_handler = logging.FileHandler(f"sm_ml_{logger_name}.log", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.INFO)
    module_logger.addHandler(file_handler)
    
    # Добавляем консольный обработчик только для КРИТИЧЕСКИХ ошибок
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))  # Минималистичный формат
    console_handler.setLevel(logging.ERROR)  # Только ERROR и CRITICAL будут выводиться в консоль
    module_logger.addHandler(console_handler)

# Специальный логгер для main.py, который будет выводить только критические сообщения в консоль
logger = logging.getLogger("ml_main")
logger.propagate = False  # Отключаем передачу логов в корневой логгер

class MLBinaryPatternAnalyzer(BinaryPatternAnalyzer):
    """
    Расширенная версия BinaryPatternAnalyzer с поддержкой ML
    """
    def __init__(self):
        super().__init__()
        
        # Инициализация ML-модулей
        self.ml_integration = MLIntegration()
        
        # Настройки ML
        self.use_ml = True  # Использовать ML-модели по умолчанию
        self.ml_weight = 0.3  # Вес ML-модели в итоговом прогнозе (0.0-1.0)
        
        # Передаем настройки в модуль интеграции
        self.ml_integration.toggle_ml(self.use_ml)
        self.ml_integration.set_ml_weight(self.ml_weight)
        
        # Запуск периодического переобучения моделей
        self.ml_integration.schedule_model_retraining(interval_hours=24)
        
        logger.info("MLBinaryPatternAnalyzer инициализирован")
    
    def toggle_ml(self, enabled=True):
        """
        Включение/отключение использования ML-моделей
        
        Args:
            enabled: использовать ML-модели (True/False)
        """
        self.use_ml = enabled
        self.ml_integration.toggle_ml(enabled)
        print(f"ML-модели {'включены' if enabled else 'отключены'}")
        logger.info(f"ML-модели {'включены' if enabled else 'отключены'}")
    
    def set_ml_weight(self, weight):
        """
        Установка веса ML-модели в итоговом прогнозе
        
        Args:
            weight: вес ML-модели (0.0-1.0)
        """
        if 0.0 <= weight <= 1.0:
            self.ml_weight = weight
            self.ml_integration.set_ml_weight(weight)
            print(f"Установлен вес ML: {weight}, вес обычных алгоритмов: {1.0 - weight}")
            logger.info(f"Установлен вес ML: {weight}, вес обычных алгоритмов: {1.0 - weight}")
        else:
            print(f"Некорректный вес ML: {weight}. Допустимый диапазон: 0.0-1.0")
            logger.warning(f"Некорректный вес ML: {weight}. Допустимый диапазон: 0.0-1.0")
    
    def analyze_binary_signal(self, candles):
        """
        Комплексный анализ для бинарных опционов с интеграцией ML
        
        Args:
            candles: список обнаруженных свечей
            
        Returns:
            tuple: (направление сигнала, уверенность, использовался ли ML)
        """
        # Получаем результат от базового метода
        direction, confidence = super().analyze_binary_signal(candles)
        
        # Если не используем ML или результат базового метода пустой, возвращаем его
        if not self.use_ml or direction is None:
            return direction, confidence, False
        
        try:
            # Получаем дополнительную информацию для ML-анализа
            conventional_results = (direction, confidence, {"signal_sources": self.signal_sources})
            
            # Применяем ML-анализ
            ml_direction, ml_confidence, ml_info, is_ml_used = self.ml_integration.analyze_pattern(candles, conventional_results)
            
            # Если ML был успешно использован, возвращаем объединенные результаты
            if is_ml_used:
                # Добавляем информацию о ML в источники сигналов
                if ml_info and "ml_confidence" in ml_info:
                    self.signal_sources += f", ML ({ml_info['ml_confidence']:.2f})"
                
                return ml_direction, ml_confidence, True
            
            # В противном случае возвращаем результат базового метода
            return direction, confidence, False
            
        except Exception as e:
            logger.error(f"Ошибка при ML-анализе сигнала: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # В случае ошибки возвращаем результат базового метода
            return direction, confidence, False
    
    def _check_prediction_result(self, forecast):
        """
        Проверка результата прогноза с обновлением ML-моделей
        Переопределяем метод из базового класса
        """
        # Вызываем оригинальный метод
        super()._check_prediction_result(forecast)
        
        # После проверки обновляем ML-модели на основе результата
        if self.use_ml and "is_correct" in forecast and forecast["is_correct"] is not None:
            try:
                # Если есть начальные данные для ML, обновляем модели
                if hasattr(self, 'ml_data') and self.ml_data:
                    pattern = forecast["pattern"]
                    is_correct = forecast["is_correct"]
                    
                    # Обновляем ML на основе обратной связи
                    self.ml_integration.update_ml_from_feedback(pattern, self.ml_data, is_correct)
                    
                    logger.info(f"ML-модель для паттерна '{pattern}' обновлена на основе обратной связи")
            except Exception as e:
                logger.error(f"Ошибка при обновлении ML-моделей: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    def _notify_binary_signal(self, action, confidence, message):
        """
        Оповещение о сигнале для бинарных опционов с сохранением ML-данных
        Переопределяем метод из базового класса
        """
        # Сохраняем данные для ML обратной связи
        if self.use_ml and self.current_candles:
            try:
                # Генерируем признаки для ML
                data_processor = MLDataProcessor()
                self.ml_data = data_processor.generate_additional_features(self.current_candles)
                logger.info(f"Сохранены данные для ML обратной связи: {len(self.ml_data)} признаков")
            except Exception as e:
                logger.error(f"Ошибка при сохранении данных для ML: {e}")
                self.ml_data = None
        
        # Вызываем оригинальный метод
        super()._notify_binary_signal(action, confidence, message)
    
    def get_ml_stats(self):
        """
        Получение статистики использования ML
        
        Returns:
            dict: статистика использования ML
        """
        return self.ml_integration.get_ml_stats()
    
    def retrain_ml_models(self):
        """
        Запуск переобучения ML-моделей
        
        Returns:
            dict: результаты переобучения
        """
        try:
            result = self.ml_integration.analyzer.train_models()
            if result["status"] == "success":
                print("ML-модели успешно переобучены")
                logger.info("ML-модели успешно переобучены")
                
                # Выводим детальную информацию по каждой модели
                for pattern, pattern_result in result.get("results", {}).items():
                    if pattern_result["status"] == "success":
                        accuracy = pattern_result.get("accuracy", 0)
                        sample_size = pattern_result.get("sample_size", 0)
                        print(f"Паттерн '{pattern}': точность {accuracy:.2f}, примеров: {sample_size}")
                        
                        # Если точность недостаточна, выводим предупреждение
                        if accuracy < 0.6:
                            print(f"Внимание: низкая точность модели для паттерна '{pattern}'")
            else:
                print(f"Проблема при переобучении ML-моделей: {result.get('message', 'Неизвестная ошибка')}")
                logger.warning(f"Проблема при переобучении ML-моделей: {result.get('message', 'Неизвестная ошибка')}")
            
            return result
            
        except Exception as e:
            error_msg = f"Ошибка при переобучении ML-моделей: {e}"
            print(error_msg)
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

def main():
    """Основная функция запуска программы с ML-возможностями"""
    # Установка кодировки вывода для Windows
    if sys.platform.startswith('win'):
        os.system('chcp 65001')  # Установка UTF-8 кодировки
        os.system('cls')
    
    # Инициализируем анализатор с ML-поддержкой
    analyzer = MLBinaryPatternAnalyzer()
    
    while True:
        print("\n===== SM-ML - ScalpMaster с искусственным интеллектом =====")
        print("1. Запустить анализ")
        print("2. Выбрать валютную пару")
        print("3. Настроить таймфрейм и время экспирации")
        print("4. Настроить параметры")
        print("5. Управление паттернами")
        print("6. Статистика прогнозов")
        print("7. Настройки ML")
        print("8. Статистика ML")
        print("9. Выход")  
        
        try:
            choice = input("\nВыберите действие: ")
            
            if choice == "1":
                analyzer.start_binary_analysis()
            elif choice == "2":
                # Выбор валютной пары
                print("\nДоступные валютные пары:")
                for i, pair in enumerate(CURRENCY_PROFILES.keys(), 1):
                    print(f"{i}. {pair}")
                
                pair_choice = input("Выберите номер пары: ")
                try:
                    pair_index = int(pair_choice) - 1
                    selected_pair = list(CURRENCY_PROFILES.keys())[pair_index]
                    analyzer.set_currency_pair(selected_pair)
                except (ValueError, IndexError):
                    print("Неверный выбор пары")
            elif choice == "3":
                # Настройка временных параметров
                print("\nНастройка таймфрейма и времени экспирации")
                print("Доступные таймфреймы: 1мин, 5мин, 15мин, 30мин")
                try:
                    timeframe = int(input("Введите таймфрейм (минуты): "))
                    
                    print("Доступное время экспирации: 1мин, 3мин, 5мин, 15мин, 30мин")
                    expiration = int(input("Введите время экспирации (минуты): "))
                    
                    analyzer.set_timeframe(timeframe)
                    analyzer.set_expiration_time(expiration)
                except ValueError:
                    print("Введите корректное числовое значение")
            elif choice == "4":
                # Настройка параметров
                print("\nНастройка параметров")
                try:
                    threshold = float(input("Введите порог уверенности (0.1-1.0): "))
                    analyzer.min_confidence_threshold = max(0.1, min(1.0, threshold))
                    
                    composite = input("Использовать комплексные сигналы? (да/нет): ").lower()
                    analyzer.use_composite_signals = composite == "да"
                    
                    print("Настройки сохранены")
                except ValueError:
                    print("Введите корректное значение для порога")
            elif choice == "5":
                # Управление паттернами
                print("\nУправление паттернами:")
                print("Доступные паттерны:")
                for i, pattern in enumerate(analyzer.pattern_names, 1):
                    enabled = pattern in analyzer.enabled_patterns
                    status = "✓" if enabled else "✗"
                    print(f"{i}. [{status}] {pattern}")
                
                print("\nВыберите действие:")
                print("1. Включить все паттерны")
                print("2. Отключить все паттерны")
                print("3. Переключить отдельный паттерн")
                print("4. Назад")
                
                pattern_choice = input("Ваш выбор: ")
                
                if pattern_choice == "1":
                    analyzer.enabled_patterns = analyzer.pattern_names.copy()
                    print("Все паттерны включены")
                elif pattern_choice == "2":
                    # Отключаем все, но оставляем хотя бы один
                    analyzer.enabled_patterns = [analyzer.pattern_names[0]]
                    print("Все паттерны отключены, кроме первого")
                elif pattern_choice == "3":
                    try:
                        idx = int(input("Введите номер паттерна: ")) - 1
                        if 0 <= idx < len(analyzer.pattern_names):
                            pattern = analyzer.pattern_names[idx]
                            enabled = pattern in analyzer.enabled_patterns
                            analyzer.toggle_pattern(pattern, not enabled)
                        else:
                            print("Неверный номер паттерна")
                    except ValueError:
                        print("Введите корректный номер")
                elif pattern_choice == "4":
                    pass
                else:
                    print("Неверный выбор")
            elif choice == "6":
                # Статистика прогнозов
                print("\nВыберите тип статистики:")
                print("1. Общая статистика прогнозов")
                print("2. Статистика по бинарным опционам")
                
                stat_choice = input("Ваш выбор (1-2): ")
                
                if stat_choice == "1":
                    # Используем тихие функции для обработки статистики
                    forecast_stats.show_forecast_stats()
                    
                    # Создаем и показываем графики только если есть статистика
                    try:
                        forecasts, stats = forecast_stats.load_pattern_forecast_data("pattern_forecast_log.csv")
                        if forecasts and len(forecasts) > 0:
                            # При наличии данных создаем графики
                            plt = forecast_stats.create_forecast_charts(stats=stats)
                            if plt:
                                plt.show()
                    except Exception as e:
                        logger.error(f"Ошибка при создании графиков: {e}")
                        
                elif stat_choice == "2":
                    show_binary_stats(analyzer.binary_log_file)
                
                input("\nНажмите Enter для продолжения...")
            elif choice == "7":
                # Настройки ML
                print("\nНастройки искусственного интеллекта:")
                print(f"1. {'Отключить' if analyzer.use_ml else 'Включить'} использование ML")
                print(f"2. Установить вес ML (текущий: {analyzer.ml_weight:.2f})")
                print("3. Переобучить ML-модели")
                print("4. Назад")
                
                ml_choice = input("Ваш выбор: ")
                
                if ml_choice == "1":
                    analyzer.toggle_ml(not analyzer.use_ml)
                elif ml_choice == "2":
                    try:
                        weight = float(input("Введите вес ML (0.0-1.0): "))
                        analyzer.set_ml_weight(weight)
                    except ValueError:
                        print("Введите корректное число")
                elif ml_choice == "3":
                    print("Переобучение ML-моделей...")
                    analyzer.retrain_ml_models()
                elif ml_choice == "4":
                    pass
                else:
                    print("Неверный выбор")
            elif choice == "8":
                # Статистика ML
                ml_stats = analyzer.get_ml_stats()
                
                print("\n===== Статистика ML =====")
                print(f"Всего прогнозов с ML: {ml_stats['total_predictions']}")
                
                if ml_stats['total_predictions'] > 0:
                    print(f"ML улучшил прогноз: {ml_stats['ml_improved']} ({ml_stats.get('ml_improved_percent', 0):.1f}%)")
                    print(f"ML ухудшил прогноз: {ml_stats['ml_worsened']} ({ml_stats.get('ml_worsened_percent', 0):.1f}%)")
                    print(f"ML нейтрален: {ml_stats['total_predictions'] - ml_stats['ml_improved'] - ml_stats['ml_worsened']} ({ml_stats.get('ml_neutral_percent', 0):.1f}%)")
                
                # Статистика по паттернам
                if ml_stats['pattern_stats']:
                    print("\nСтатистика по паттернам:")
                    print("-" * 60)
                    print(f"{'Паттерн':<20} {'Всего':<8} {'ML улучшил':<12} {'ML ухудшил':<12} {'Уверенность ML':<15}")
                    print("-" * 60)
                    
                    for pattern, pattern_stats in ml_stats['pattern_stats'].items():
                        total = pattern_stats['total']
                        improved = pattern_stats.get('ml_improved', 0)
                        worsened = pattern_stats.get('ml_worsened', 0)
                        avg_ml_conf = pattern_stats.get('avg_ml_confidence', 0)
                        
                        print(f"{pattern:<20} {total:<8} {improved:<12} {worsened:<12} {avg_ml_conf:.2f}")
                
                input("\nНажмите Enter для продолжения...")
            elif choice == "9":
                print("Выход из программы.")
                break
            else:
                print("Неверный выбор.")
        except Exception as e:
            logger.error(f"Произошла ошибка: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print("Пожалуйста, попробуйте снова.")

def show_binary_stats(log_file):
    """
    Отображение статистики бинарных опционов
    """
    try:
        # Добавляем необходимые импорты
        import csv
        import os
        
        # Проверяем существование файла
        if not os.path.exists(log_file):
            print(f"Файл {log_file} не найден")
            return
        
        # Пробуем прочитать файл вручную
        win_count = 0
        loss_count = 0
        total_pnl = 0
        pair_stats = {}
        
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Пропускаем заголовок
            
            for row in reader:
                if len(row) < 8:  # Проверяем, достаточно ли столбцов
                    continue
                
                pair = row[1]  # Валютная пара
                result = row[7]  # Результат (WIN/LOSS)
                
                # Обновляем статистику по паре
                if pair not in pair_stats:
                    pair_stats[pair] = {"total": 0, "wins": 0}
                
                pair_stats[pair]["total"] += 1
                
                if result == "WIN":
                    win_count += 1
                    pair_stats[pair]["wins"] += 1
                elif result == "LOSS":
                    loss_count += 1
                
                # Обрабатываем P&L если есть
                if len(row) >= 9 and row[8]:
                    try:
                        pnl = float(row[8])
                        total_pnl += pnl
                    except ValueError:
                        pass
        
        # Выводим статистику по парам
        total_trades = win_count + loss_count
        print("\nСтатистика по валютным парам:")
        print("-" * 60)
        print(f"{'Пара':<10} {'Сделок':<10} {'Побед':<10} {'Процент':<10}")
        print("-" * 60)
        
        for pair, stats in pair_stats.items():
            if stats["total"] > 0:
                win_percent = (stats["wins"] / stats["total"]) * 100
                print(f"{pair:<10} {stats['total']:<10} {stats['wins']:<10} {win_percent:.1f}%")
        
        print("-" * 60)
        
        # Общая статистика
        if total_trades > 0:
            win_percent = (win_count / total_trades) * 100
            print(f"\nОбщая статистика: {win_count}/{total_trades} ({win_percent:.1f}%)")
            print(f"Общая прибыль/убыток: {total_pnl:.2f}")
        else:
            print("\nНет завершенных сделок для расчета статистики")
    
    except Exception as e:
        logger.error(f"Ошибка при отображении статистики: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nПрограмма завершена пользователем.")
    except Exception as e:
        logger.error(f"\nКритическая ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        input("Нажмите Enter для выхода...")