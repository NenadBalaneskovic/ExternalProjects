"""
config.py
Central configuration for Weather Aggregator.
Stores API keys, endpoints, and model registry.
"""

class Config:
    # API Keys
    WEATHER_API_KEY = "Insert your OpenWeatherMap API-Key"
    HF_API_TOKEN = "Insert your HuggingFace API-Key"

    # API URLs
    HF_API_URL = "https://api-inference.huggingface.co/models/"

    # Model Registry (extendable)
    MODEL_REGISTRY = {
        "SARIMAX": {"order": (1,1,1)},
        "RandomForest": {"n_estimators": 50},
        "Kalman": {"window": 3},
        "ridge": {"alpha": 1.0},
        "logistic": {},
        "boosting": {"n_estimators": 100}
    }
