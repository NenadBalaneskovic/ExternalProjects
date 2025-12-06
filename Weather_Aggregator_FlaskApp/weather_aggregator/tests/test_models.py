"""
test_models.py
Unit tests for base models and explainability.
"""

import numpy as np
from models import base_models, explainability

def test_sarimax_forecast():
    series = np.random.rand(50)
    forecast = base_models.sarimax_forecast(series, steps=3)
    assert len(forecast) == 3

def test_kalman_forecast():
    series = np.random.rand(10)
    forecast = base_models.kalman_filter_forecast(series, steps=2)
    assert len(forecast) == 2

def test_random_forest_forecast():
    series = np.random.rand(20)
    forecast = base_models.random_forest_forecast(series, steps=4)
    assert len(forecast) == 4

def test_explainability_shap():
    series = np.random.rand(20)
    X = series.reshape(-1,1)
    exp = explainability.Explainability(X)
    # SHAP analysis should return values without error
    shap_values = exp.shap_analysis(lambda x: x, X[:1])
    assert shap_values is not None
