"""
Файл конфигурации для SM (ScalpMaster)
"""

import json
import os

# Имя файла конфигурации
CONFIG_FILE = "sm_config.json"

# Настройки по умолчанию
DEFAULT_CONFIG = {
    "currency_pair": "EUR/USD",
    "timeframe": 1,
    "expiration_time": 3,
    "min_confidence_threshold": 0.6,
    "use_composite_signals": True,
    "signal_weights": {
        "patterns": 0.4,
        "indicators": 0.4,
        "momentum": 0.2
    },
    "monitor_area": {
        "top": 100,
        "left": 100,
        "width": 800,
        "height": 600
    },
    "sound_enabled": True,
    "auto_screenshot": False,
    "auto_trading": False
}

def load_config():
    """
    Загрузка конфигурации из файла
    
    Returns:
        dict: загруженная конфигурация или настройки по умолчанию
    """
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"Конфигурация загружена из {CONFIG_FILE}")
                return config
        else:
            print(f"Файл конфигурации не найден, используются настройки по умолчанию")
            save_config(DEFAULT_CONFIG)  # Сохраняем настройки по умолчанию
            return DEFAULT_CONFIG
    except Exception as e:
        print(f"Ошибка при загрузке конфигурации: {e}")
        return DEFAULT_CONFIG

def save_config(config):
    """
    Сохранение конфигурации в файл
    
    Args:
        config: словарь с настройками для сохранения
    """
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
            print(f"Конфигурация сохранена в {CONFIG_FILE}")
    except Exception as e:
        print(f"Ошибка при сохранении конфигурации: {e}")

def update_config(key, value):
    """
    Обновление одного параметра конфигурации
    
    Args:
        key: ключ параметра
        value: новое значение
    """
    config = load_config()
    config[key] = value
    save_config(config)
    print(f"Параметр {key} обновлен: {value}")