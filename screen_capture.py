"""
Модуль для захвата экрана в системе ScalpMaster (SM)
Отвечает за выбор области мониторинга и захват изображений
"""

import cv2
import numpy as np
import time
import logging

# Настройка логирования
logger = logging.getLogger("screen_capture")

def capture_screen(area):
    """
    Захват указанной области экрана
    
    Args:
        area: словарь с координатами области {top, left, width, height, monitor}
        
    Returns:
        np.array: изображение в формате BGR
    """
    try:
        import mss
        with mss.mss() as sct:
            screenshot = np.array(sct.grab(area))
            return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    except Exception as e:
        logger.error(f"Ошибка при захвате экрана: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)

def select_monitor_area(monitor_number=None):
    """
    Позволяет пользователю выбрать область экрана для мониторинга
    
    Args:
        monitor_number: номер монитора (None - выбор монитора)
    
    Returns:
        dict: область экрана {top, left, width, height} или None при отмене
    """
    logger.info("Выбор монитора и области для анализа...")
    
    try:
        import mss
        
        with mss.mss() as sct:
            # Показываем доступные мониторы
            if monitor_number is None:
                logger.info("Доступные мониторы:")
                for i, monitor in enumerate(sct.monitors[1:], 1):  # Пропускаем первый (объединенный) монитор
                    logger.info(f"{i}. Монитор {i}: {monitor['width']}x{monitor['height']}")
                
                monitor_choice = input("Выберите номер монитора (или Enter для основного): ")
                if monitor_choice.strip():
                    monitor_number = int(monitor_choice)
                else:
                    monitor_number = 1  # Основной монитор
            
            # Получаем информацию о выбранном мониторе
            monitor_info = sct.monitors[monitor_number]
            
            # Захватываем весь монитор для выбора области
            screen = np.array(sct.grab(monitor_info))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
            
            # Закрываем все окна перед созданием нового
            cv2.destroyAllWindows()
            time.sleep(0.5)  # Даем время на закрытие окон
            
            cv2.namedWindow("Select Area", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Select Area", 800, 600)
            
            # Инициализируем переменные
            roi_points = []
            
            def mouse_callback(event, x, y, flags, param):
                nonlocal screen, roi_points
                img_copy = screen.copy()  # Создаем копию оригинального изображения
                
                if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 2:
                    roi_points.append((x, y))
                    
                    # Отображаем выбранные точки
                    for point in roi_points:
                        cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
                    
                    # Если выбраны обе точки, рисуем прямоугольник
                    if len(roi_points) == 2:
                        x1, y1 = roi_points[0]
                        x2, y2 = roi_points[1]
                        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    cv2.imshow("Select Area", img_copy)
            
            logger.info("Кликните в двух противоположных углах области, которую вы хотите проанализировать.")
            logger.info("Нажмите ESC для выхода.")
            
            cv2.setMouseCallback("Select Area", mouse_callback)
            
            # Показываем начальное изображение
            cv2.imshow("Select Area", screen)
            
            while len(roi_points) < 2:
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC для выхода
                    cv2.destroyAllWindows()
                    return None
            
            # Вычисляем координаты области
            x1, y1 = min(roi_points[0][0], roi_points[1][0]), min(roi_points[0][1], roi_points[1][1])
            x2, y2 = max(roi_points[0][0], roi_points[1][0]), max(roi_points[0][1], roi_points[1][1])
            
            # Создаем область относительно выбранного монитора
            monitor_area = {
                "top": y1 + monitor_info["top"],  # Добавляем смещение монитора
                "left": x1 + monitor_info["left"],  # Добавляем смещение монитора
                "width": x2 - x1,
                "height": y2 - y1,
                "monitor": monitor_number  # Запоминаем номер монитора
            }
            
            cv2.destroyAllWindows()
            
            logger.info(f"Выбрана область: {monitor_area}")
            return monitor_area
    except Exception as e:
        logger.error(f"Ошибка при выборе области: {e}")
        import traceback
        logger.error(traceback.format_exc())
        cv2.destroyAllWindows()
        return None

def take_screenshot(frame, prefix="sm"):
    """
    Сохранение скриншота в файл
    
    Args:
        frame: изображение для сохранения
        prefix: префикс имени файла
        
    Returns:
        str: имя созданного файла
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{prefix}_screenshot_{timestamp}.png"
    cv2.imwrite(filename, frame)
    logger.info(f"Скриншот сохранен: {filename}")
    return filename
