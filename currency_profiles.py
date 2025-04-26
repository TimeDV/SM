"""
Профили валютных пар для системы ScalpMaster (SM)
Содержит параметры и настройки для разных валютных пар
"""

CURRENCY_PROFILES = {
    "EUR/USD": {
        "volatility": "medium", 
        "avg_range_1m": 0.0004,
        "pattern_sensitivity": {
            "double_bottom": 0.22,
            "double_top": 0.22,
            "ascending_wedge": 0.25,
            "descending_wedge": 0.25
        },
        "rsi_levels": {
            "overbought": 70,
            "oversold": 30
        }
    },
    "EUR/JPY": {
        "volatility": "high",
        "avg_range_1m": 0.06,
        "pattern_sensitivity": {
            "double_bottom": 0.25,
            "double_top": 0.25,
            "ascending_wedge": 0.28,
            "descending_wedge": 0.28
        },
        "rsi_levels": {
            "overbought": 75,
            "oversold": 25
        }
    },
    "EUR/CAD": {
        "volatility": "medium",
        "avg_range_1m": 0.0005,
        "pattern_sensitivity": {
            "double_bottom": 0.22,
            "double_top": 0.22,
            "ascending_wedge": 0.25,
            "descending_wedge": 0.25
        },
        "rsi_levels": {
            "overbought": 70,
            "oversold": 30
        }
    },
    "CAD/JPY": {
        "volatility": "high",
        "avg_range_1m": 0.055,
        "pattern_sensitivity": {
            "double_bottom": 0.25,
            "double_top": 0.25,
            "ascending_wedge": 0.28,
            "descending_wedge": 0.28
        },
        "rsi_levels": {
            "overbought": 75,
            "oversold": 25
        }
    },
    "NZD/CAD": {
        "volatility": "medium-low",
        "avg_range_1m": 0.0004,
        "pattern_sensitivity": {
            "double_bottom": 0.20,
            "double_top": 0.20,
            "ascending_wedge": 0.23,
            "descending_wedge": 0.23
        },
        "rsi_levels": {
            "overbought": 68,
            "oversold": 32
        }
    }
}

# Функция получения профиля валютной пары
def get_currency_profile(pair):
    """
    Получить профиль для указанной валютной пары
    
    Args:
        pair: код валютной пары (например, 'EUR/USD')
        
    Returns:
        dict: профиль валютной пары или None, если не найден
    """
    return CURRENCY_PROFILES.get(pair, None)