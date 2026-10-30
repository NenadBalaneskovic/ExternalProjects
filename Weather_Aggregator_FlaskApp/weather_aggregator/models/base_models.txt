"""
base_models.py
Classical + ML forecasting models for Weather Aggregator.
Includes SARIMAX, Kalman filter, Trees, CNNs, LSTMs, Autoencoders.
"""

import logging
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

logger = logging.getLogger(__name__)


# --- Helper to extract numeric series ---
def _extract_series(weather_data, key="temperature_series"):
    """
    Extract numeric temperature series from weather_data dict.
    Falls back to single temperature if no series is available.
    """
    if weather_data is None:
        return np.array([])

    # If API provides a time series (e.g. hourly temps)
    if key in weather_data and isinstance(weather_data[key], (list, np.ndarray)):
        return np.array(weather_data[key], dtype=float)

    # Fallback: use single temperature value
    if "temperature" in weather_data:
        return np.array([float(weather_data["temperature"])])

    return np.array([])


# --- Classical Models ---

def sarimax_forecast(weather_data, steps=5):
    """SARIMAX forecast."""
    series = _extract_series(weather_data)
    if len(series) < 3:
        logger.warning("Not enough data for SARIMAX. Returning average.")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])

    try:
        model = sm.tsa.SARIMAX(series, order=(1, 1, 1))
        results = model.fit(disp=False)
        forecast = results.forecast(steps=steps)
        return forecast
    except Exception as e:
        logger.exception(f"SARIMAX forecast failed: {e}")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])


def kalman_filter_forecast(weather_data, steps=5):
    """Simple Kalman filter forecast (rolling mean proxy)."""
    series = _extract_series(weather_data)
    if len(series) == 0:
        return np.array([])

    forecast = [np.mean(series[-3:])] * steps
    return np.array(forecast)


# --- Tree-Based Models ---

def random_forest_forecast(weather_data, steps=5):
    """Random Forest regression forecast."""
    series = _extract_series(weather_data)
    if len(series) < 2:
        logger.warning("Not enough data for RandomForest. Returning average.")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])

    try:
        X = np.arange(len(series)).reshape(-1, 1)
        y = series
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X, y)
        future = np.arange(len(series), len(series) + steps).reshape(-1, 1)
        return rf.predict(future)
    except Exception as e:
        logger.exception(f"RandomForest forecast failed: {e}")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])


def gradient_boosting_forecast(weather_data, steps=5):
    """Gradient Boosting regression forecast."""
    series = _extract_series(weather_data)
    if len(series) < 2:
        logger.warning("Not enough data for GradientBoosting. Returning average.")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])

    try:
        X = np.arange(len(series)).reshape(-1, 1)
        y = series
        gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
        gb.fit(X, y)
        future = np.arange(len(series), len(series) + steps).reshape(-1, 1)
        return gb.predict(future)
    except Exception as e:
        logger.exception(f"GradientBoosting forecast failed: {e}")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])


# --- Neural Models (stubs) ---

def lstm_forecast(weather_data, steps=5):
    """LSTM forecast stub."""
    series = _extract_series(weather_data)
    if len(series) < 2:
        logger.warning("Not enough data for LSTM. Returning average.")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])

    try:
        X = np.array(series).reshape(-1, 1, 1)
        model = Sequential([
            LSTM(10, input_shape=(1, 1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, series, epochs=5, verbose=0)
        preds = model.predict(np.array(series[-steps:]).reshape(-1, 1, 1))
        return preds.flatten()
    except Exception as e:
        logger.exception(f"LSTM forecast failed: {e}")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])


def cnn_forecast(weather_data, steps=5):
    """CNN forecast stub."""
    series = _extract_series(weather_data)
    if len(series) < 2:
        logger.warning("Not enough data for CNN. Returning average.")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])

    try:
        X = np.array(series).reshape(-1, 1, 1)
        model = Sequential([
            Conv1D(8, kernel_size=1, activation="relu", input_shape=(1, 1)),
            Flatten(),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, series, epochs=5, verbose=0)
        preds = model.predict(np.array(series[-steps:]).reshape(-1, 1, 1))
        return preds.flatten()
    except Exception as e:
        logger.exception(f"CNN forecast failed: {e}")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])


def autoencoder_forecast(weather_data, steps=5):
    """Autoencoder forecast stub (reconstruction-based)."""
    series = _extract_series(weather_data)
    if len(series) < 2:
        logger.warning("Not enough data for Autoencoder. Returning average.")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])

    try:
        X = np.array(series).reshape(-1, 1)
        model = Sequential([
            Dense(5, activation="relu", input_shape=(1,)),
            Dense(1, activation="linear")
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, series, epochs=5, verbose=0)
        preds = model.predict(np.array(series[-steps:]).reshape(-1, 1))
        return preds.flatten()
    except Exception as e:
        logger.exception(f"Autoencoder forecast failed: {e}")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])
