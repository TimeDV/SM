"""
Запуск усовершенствованной версии ScalpMaster с LSTM и расширенными признаками
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

# Импорт усовершенствованных ML-модулей
from ml_enhanced_integration import EnhancedMLIntegration
from lstm_analyzer import LSTMAnalyzer 
from advanced_features import AdvancedFeatureExtractor
from ml_visualization import MLVisualizer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler("sm_enhanced_app.log", encoding='utf-8')
    ]
)

# Отключаем вывод в консоль логов с уровнем INFO и ниже от всех модулей
for logger_name in ['screen_capture', 'candle_detection', 'pattern_detection', 
                    'visualization', 'binary_analyzer', 'simple_analyzer', 
                    'forecast_stats', 'lstm_analyzer', 'ml_enhanced_integration',
                    'advanced_features', 'ml_visualization']:
    module_logger = logging.getLogger(logger_name)
    module_logger.propagate = False  # Отключаем передачу логов в корневой логгер
    
    # Добавляем файловый обработчик для сохранения всех логов в файл
    file_handler = logging.FileHandler(f"sm_enhanced_{logger_name}.log", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.INFO)
    module_logger.addHandler(file_handler)
    
    # Добавляем консольный обработчик только для КРИТИЧЕСКИХ ошибок
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))  # Минималистичный формат
    console_handler.setLevel(logging.ERROR)  # Только ERROR и CRITICAL будут выводиться в консоль
    module_logger.addHandler(console_handler)

# Специальный логгер для main.py, который будет выводить только критические сообщения в консоль
logger = logging.getLogger("enhanced_main")
logger.propagate = False  # Отключаем передачу логов в корневой логгер

class EnhancedBinaryPatternAnalyzer(BinaryPatternAnalyzer):
    """
    Расширенная версия BinaryPatternAnalyzer с поддержкой усовершенствованной ML
    """
    def __init__(self):
        super().__init__()
        
        # Инициализация усовершенствованных ML-модулей
        self.ml_integration = EnhancedMLIntegration()
        self.visualizer = MLVisualizer()
        
        # Настройки ML
        self.use_ml = True  # Использовать ML-модели по умолчанию
        self.ml_weight = 0.3  # Вес ML-модели в итоговом прогнозе (0.0-1.0)
        self.use_lstm = True  # Использовать LSTM по умолчанию
        self.lstm_weight = 0.3  # Вес LSTM-модели в итоговом прогнозе (0.0-1.0)
        
        # Передаем настройки в модуль интеграции
        self.ml_integration.toggle_ml(self.use_ml)
        self.ml_integration.set_ml_weight(self.ml_weight)
        self.ml_integration.toggle_lstm(self.use_lstm)
        self.ml_integration.set_lstm_weight(self.lstm_weight)
        
        # Запуск периодического переобучения моделей
        self.ml_integration.schedule_model_retraining(interval_hours=24)
        self.ml_integration.schedule_lstm_retraining(interval_hours=24)
        
        logger.info("EnhancedBinaryPatternAnalyzer инициализирован")
    
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
    
    def toggle_lstm(self, enabled=True):
        """
        Включение/отключение использования LSTM
        
        Args:
            enabled: использовать LSTM (True/False)
        """
        self.use_lstm = enabled
        self.ml_integration.toggle_lstm(enabled)
        print(f"LSTM {'включен' if enabled else 'отключен'}")
        logger.info(f"LSTM {'включен' if enabled else 'отключен'}")
    
    def set_ml_weight(self, weight):
        """
        Установка веса ML-модели в итоговом прогнозе
        
        Args:
            weight: вес ML-модели (0.0-1.0)
        """
        if 0.0 <= weight <= 1.0:
            self.ml_weight = weight
            self.ml_integration.set_ml_weight(weight)
            print(f"Установлен вес ML: {weight}")
            logger.info(f"Установлен вес ML: {weight}")
        else:
            print(f"Некорректный вес ML: {weight}. Допустимый диапазон: 0.0-1.0")
            logger.warning(f"Некорректный вес ML: {weight}. Допустимый диапазон: 0.0-1.0")
    
    def set_lstm_weight(self, weight):
        """
        Установка веса LSTM в итоговом прогнозе
        
        Args:
            weight: вес LSTM (0.0-1.0)
        """
        if 0.0 <= weight <= 1.0:
            self.lstm_weight = weight
            self.ml_integration.set_lstm_weight(weight)
            print(f"Установлен вес LSTM: {weight}")
            logger.info(f"Установлен вес LSTM: {weight}")
        else:
            print(f"Некорректный вес LSTM: {weight}. Допустимый диапазон: 0.0-1.0")
            logger.warning(f"Некорректный вес LSTM: {weight}. Допустимый диапазон: 0.0-1.0")
    
    def analyze_binary_signal(self, candles):
        """
        Комплексный анализ для бинарных опционов с интеграцией усовершенствованной ML
        
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
            
            # Применяем усовершенствованный ML-анализ
            ml_direction, ml_confidence, ml_info, is_enhanced = self.ml_integration.analyze_pattern(
                candles, 
                conventional_results,
                currency_pair=self.current_pair
            )
            
            # Если усовершенствованная ML была успешно использована, возвращаем объединенные результаты
            if is_enhanced:
                # Добавляем информацию о ML в источники сигналов
                self.signal_sources += f", Enhanced ML ({ml_confidence:.2f})"
                
                # Добавляем прогноз в историю для визуализации
                prediction_data = {
                    'timestamp': datetime.now(),
                    'pair': self.current_pair,
                    'direction': ml_direction,
                    'confidence': ml_confidence,
                    'conventional_confidence': confidence,
                    'ml_confidence': ml_info.get('ml_confidence', 0),
                    'lstm_confidence': ml_info.get('lstm_confidence', 0),
                    'result': None,  # Будет заполнено при проверке
                    'market_session': ml_info.get('market_session', 'unknown'),
                    'noise_level': ml_info.get('noise_level', 0)
                }
                self.visualizer.add_prediction_to_history(prediction_data)
                
                # Сохраняем информацию о прогнозе для последующей проверки
                self.current_prediction = prediction_data
                
                return ml_direction, ml_confidence, True
            
            # В противном случае возвращаем результат базового метода
            return direction, confidence, False
            
        except Exception as e:
            logger.error(f"Ошибка при усовершенствованном ML-анализе сигнала: {e}")
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
        
        # После проверки обновляем текущий прогноз в истории визуализатора
        try:
            # Если есть текущий прогноз и его результат был определен
            if hasattr(self, 'current_prediction') and forecast["is_correct"] is not None:
                # Обновляем результат прогноза
                self.current_prediction['result'] = forecast["is_correct"]
                
                # Обновляем историю прогнозов
                self.visualizer.save_predictions_history()
                
                # После результата генерируем или обновляем отчет
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
                    # Вызываем функцию показа статистики по бинарным опционам
                    import main
                    main.show_binary_stats(analyzer.binary_log_file)
                
                input("\nНажмите Enter для продолжения...")
            elif choice == "7":
                # Настройки ML
                print("\nНастройки искусственного интеллекта:")
                print(f"1. {'Отключить' if analyzer.use_ml else 'Включить'} ML")
                print(f"2. {'Отключить' if analyzer.use_lstm else 'Включить'} LSTM")
                print(f"3. Установить вес ML (текущий: {analyzer.ml_weight:.2f})")
                print(f"4. Установить вес LSTM (текущий: {analyzer.lstm_weight:.2f})")
                print("5. Переобучить ML-модели")
                print("6. Назад")
                
                ml_choice = input("Ваш выбор: ")
                
                if ml_choice == "1":
                    analyzer.toggle_ml(not analyzer.use_ml)
                elif ml_choice == "2":
                    analyzer.toggle_lstm(not analyzer.use_lstm)
                elif ml_choice == "3":
                    try:
                        weight = float(input("Введите вес ML (0.0-1.0): "))
                        analyzer.set_ml_weight(weight)
                    except ValueError:
                        print("Введите корректное число")
                elif ml_choice == "4":
                    try:
                        weight = float(input("Введите вес LSTM (0.0-1.0): "))
                        analyzer.set_lstm_weight(weight)
                    except ValueError:
                        print("Введите корректное число")
                elif ml_choice == "5":
                    print("Переобучение ML-моделей...")
                    # Здесь должен быть код переобучения, но на данном этапе
                    # это может быть просто заглушка
                    print("Функция переобучения требует исторических данных")
                    print("В текущей версии она недоступна")
                elif ml_choice == "6":
                    pass
                else:
                    print("Неверный выбор")
            elif choice == "8":
                # Статистика ML
                enhanced_stats = analyzer.get_enhanced_stats()
                
                print("\n===== Расширенная статистика ML =====")
                print(f"Всего прогнозов с ML: {enhanced_stats.get('total_predictions', 0)}")
                
                # Выводим статистику ML-моделей
                if 'total_predictions' in enhanced_stats and enhanced_stats['total_predictions'] > 0:
                    print(f"ML улучшил прогноз: {enhanced_stats.get('ml_improved', 0)} ({enhanced_stats.get('ml_improved_percent', 0):.1f}%)")
                    print(f"ML ухудшил прогноз: {enhanced_stats.get('ml_worsened', 0)} ({enhanced_stats.get('ml_worsened_percent', 0):.1f}%)")
                    print(f"ML нейтрален: {enhanced_stats['total_predictions'] - enhanced_stats.get('ml_improved', 0) - enhanced_stats.get('ml_worsened', 0)} ({enhanced_stats.get('ml_neutral_percent', 0):.1f}%)")
                
                # Выводим статистику LSTM
                if 'lstm' in enhanced_stats:
                    lstm_stats = enhanced_stats['lstm']
                    print("\nСтатистика LSTM:")
                    print(f"Всего прогнозов с LSTM: {lstm_stats.get('total_predictions', 0)}")
                    
                    if lstm_stats.get('total_predictions', 0) > 0:
                        print(f"LSTM улучшил прогноз: {lstm_stats.get('lstm_improved', 0)} ({lstm_stats.get('improved_percent', 0):.1f}%)")
                        print(f"LSTM ухудшил прогноз: {lstm_stats.get('lstm_worsened', 0)} ({lstm_stats.get('worsened_percent', 0):.1f}%)")
                
                # Статистика по паттернам
                if 'pattern_stats' in enhanced_stats:
                    print("\nСтатистика по паттернам:")
                    print("-" * 60)
                    print(f"{'Паттерн':<20} {'Всего':<8} {'ML улучшил':<12} {'ML ухудшил':<12} {'Уверенность ML':<15}")
                    print("-" * 60)
                    
                    for pattern, pattern_stats in enhanced_stats.get('pattern_stats', {}).items():
                        total = pattern_stats.get('total', 0)
                        improved = pattern_stats.get('ml_improved', 0)
                        worsened = pattern_stats.get('ml_worsened', 0)
                        avg_ml_conf = pattern_stats.get('avg_ml_confidence', 0)
                        
                        print(f"{pattern:<20} {total:<8} {improved:<12} {worsened:<12} {avg_ml_conf:.2f}")
                
                input("\nНажмите Enter для продолжения...")
            elif choice == "9":
                # Генерировать отчет ML
                print("Генерация отчета о производительности ML...")
                report_path = analyzer.generate_visual_report()
                
                if report_path:
                    print(f"Отчет успешно сгенерирован: {report_path}")
                else:
                    print("Не удалось сгенерировать отчет (недостаточно данных)")
                
                input("\nНажмите Enter для продолжения...")
            elif choice == "0":
                print("Выход из программы.")
                break
            else:
                print("Неверный выбор.")
        except Exception as e:
            logger.error(f"Произошла ошибка: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print("Пожалуйста, попробуйте снова.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nПрограмма завершена пользователем.")
    except Exception as e:
        logger.error(f"\nКритическая ошибка: {e}")
        import traceback
        logger.error(traceback.format_exc())
        input("Нажмите Enter для выхода...") hasattr(self, 'report_counter'):
                    self.report_counter += 1
                    # Генерируем отчет каждые 10 прогнозов
                    if self.report_counter >= 10:
                        self.visualizer.generate_comprehensive_report()
                        self.report_counter = 0
                else:
                    self.report_counter = 1
        except Exception as e:
            logger.error(f"Ошибка при обновлении прогноза в истории: {e}")
        
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
                feature_extractor = AdvancedFeatureExtractor()
                self.ml_data = feature_extractor.calculate_advanced_features(
                    self.current_candles,
                    pair=self.current_pair
                )
                logger.info(f"Сохранены данные для ML обратной связи: {len(self.ml_data)} признаков")
            except Exception as e:
                logger.error(f"Ошибка при сохранении данных для ML: {e}")
                self.ml_data = None
        
        # Вызываем оригинальный метод
        super()._notify_binary_signal(action, confidence, message)
    
    def get_enhanced_stats(self):
        """
        Получение расширенной статистики использования ML
        
        Returns:
            dict: расширенная статистика
        """
        return self.ml_integration.get_enhanced_stats()
    
    def generate_visual_report(self):
        """
        Генерация визуального отчета о производительности ML
        
        Returns:
            str: путь к сгенерированному отчету
        """
        return self.visualizer.generate_comprehensive_report()

def main():
    """Основная функция запуска программы с расширенной ML-функциональностью"""
    # Установка кодировки вывода для Windows
    if sys.platform.startswith('win'):
        os.system('chcp 65001')  # Установка UTF-8 кодировки
        os.system('cls')
    
    # Инициализируем анализатор с усовершенствованной ML-поддержкой
    analyzer = EnhancedBinaryPatternAnalyzer()
    
    while True:
        print("\n===== SM-Enhanced - ScalpMaster с усовершенствованным искусственным интеллектом =====")
        print("1. Запустить анализ")
        print("2. Выбрать валютную пару")
        print("3. Настроить таймфрейм и время экспирации")
        print("4. Настроить параметры")
        print("5. Управление паттернами")
        print("6. Статистика прогнозов")
        print("7. Настройки ML")
        print("8. Статистика ML")
        print("9. Генерировать отчет ML")
        print("0. Выход")  
        
        try:
            choice = input("\nВыберите действие: ")
            
            # ... [обработка различных вариантов выбора]
            
        except Exception as e:
            logger.error(f"Произошла ошибка: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print("Пожалуйста, попробуйте снова.")