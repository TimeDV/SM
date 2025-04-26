"""
Главный файл системы SM (ScalpMaster) - анализатор для бинарных опционов.
Запускает пользовательский интерфейс и инициализирует анализатор.
"""

import logging
import os
import sys

import forecast_stats
from binary_analyzer import BinaryPatternAnalyzer
from currency_profiles import CURRENCY_PROFILES

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler("sm_app.log", encoding='utf-8')
    ]
)

# Отключаем вывод в консоль логов с уровнем INFO и ниже от всех модулей
for logger_name in ['screen_capture', 'candle_detection', 'pattern_detection', 
                    'visualization', 'binary_analyzer', 'simple_analyzer', 
                    'forecast_stats']:
    module_logger = logging.getLogger(logger_name)
    module_logger.propagate = False  # Отключаем передачу логов в корневой логгер
    
    # Добавляем файловый обработчик для сохранения всех логов в файл
    file_handler = logging.FileHandler(f"sm_{logger_name}.log", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.INFO)
    module_logger.addHandler(file_handler)
    
    # Добавляем консольный обработчик только для КРИТИЧЕСКИХ ошибок
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))  # Минималистичный формат
    console_handler.setLevel(logging.ERROR)  # Только ERROR и CRITICAL будут выводиться в консоль
    module_logger.addHandler(console_handler)

# Специальный логгер для main.py, который будет выводить только критические сообщения в консоль
logger = logging.getLogger("main")
logger.propagate = False  # Отключаем передачу логов в корневой логгер

def main():
    """Основная функция запуска программы"""
    # Установка кодировки вывода для Windows
    if sys.platform.startswith('win'):
        os.system('chcp 65001')  # Установка UTF-8 кодировки
        os.system('cls')
    
    analyzer = BinaryPatternAnalyzer()
    
    while True:
        print("\n===== SM - ScalpMaster для бинарных опционов =====")
        print("1. Запустить анализ")
        print("2. Выбрать валютную пару")
        print("3. Настроить таймфрейм и время экспирации")
        print("4. Настроить параметры")
        print("5. Управление паттернами")
        print("6. Статистика прогнозов")
        print("7. Выход")  
        
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
                timeframe = int(input("Введите таймфрейм (минуты): "))
                
                print("Доступное время экспирации: 1мин, 3мин, 5мин, 15мин, 30мин")
                expiration = int(input("Введите время экспирации (минуты): "))
                
                analyzer.set_timeframe(timeframe)
                analyzer.set_expiration_time(expiration)
            elif choice == "4":
                # Настройка параметров
                print("\nНастройка параметров")
                threshold = float(input("Введите порог уверенности (0.1-1.0): "))
                analyzer.min_confidence_threshold = max(0.1, min(1.0, threshold))
                
                composite = input("Использовать комплексные сигналы? (да/нет): ").lower()
                analyzer.use_composite_signals = composite == "да"
                
                print("Настройки сохранены")
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
                    except Exception:
                        pass
                        
                elif stat_choice == "2":
                    show_binary_stats(analyzer.binary_log_file)
                
                input("\nНажмите Enter для продолжения...")
                
            elif choice == "7":
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
        
        # Пробуем прочитать файл вручную, без pandas
        
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