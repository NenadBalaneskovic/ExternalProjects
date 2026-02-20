# analyzer/modules/forecasting.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.linear_model import LinearRegression


def run(df: pd.DataFrame) -> Tuple[Dict[str, Any], Optional[Dict[str, str]]]:
    """
    Lightweight forecasting module for real-time telemetry.

    Responsibilities:
        - Select a primary numeric column
        - Fit a simple linear regression on the last N points
        - Predict short-term future values
        - Detect unstable trends (health warnings)
        - Return:
            (1) Visualization-ready forecast data
            (2) Optional health summary

    Returns:
        result: dict
            {
                "history_x": [...],
                "history_y": [...],
                "forecast_x": [...],
                "forecast_y": [...]
            }

        health: dict | None
    """

    if df is None or df.empty:
        return {}, None

    # ---------------------------------------------------------
    # 1. Select numeric column
    # ---------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {}, None

    col = numeric_cols[0]
    y = df[col].values
    n = len(y)

    # Need at least 5 points for a meaningful trend
    if n < 5:
        return {}, None

    # ---------------------------------------------------------
    # 2. Prepare regression data
    # ---------------------------------------------------------
    x = np.arange(n).reshape(-1, 1)
    model = LinearRegression()

    try:
        model.fit(x, y)
    except Exception:
        return {}, None

    # ---------------------------------------------------------
    # 3. Forecast next 20 steps
    # ---------------------------------------------------------
    horizon = 20
    forecast_x = np.arange(n, n + horizon).reshape(-1, 1)
    forecast_y = model.predict(forecast_x)

    # ---------------------------------------------------------
    # 4. Health evaluation
    # ---------------------------------------------------------
    health = None

    slope = float(model.coef_[0])
    if abs(slope) > 5 * np.std(y):
        health = {
            "status": "Warning",
            "message": f"Unstable trend detected in {col} (slope={slope:.2f})"
        }

    # Forecast explosion
    if np.any(np.abs(forecast_y) > 1e6):
        health = {
            "status": "Error",
            "message": f"Forecast values for {col} are unrealistic"
        }

    # ---------------------------------------------------------
    # 5. Visualization output
    # ---------------------------------------------------------
    result = {
        "history_x": x.flatten().tolist(),
        "history_y": y.tolist(),
        "forecast_x": forecast_x.flatten().tolist(),
        "forecast_y": forecast_y.tolist(),
        "label": col,
    }

    return result, health