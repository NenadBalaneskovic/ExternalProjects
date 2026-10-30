# analyzer/modules/statistics.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional


def run(df: pd.DataFrame) -> Tuple[Dict[str, Any], Optional[Dict[str, str]]]:
    """
    Runs lightweight statistical analysis on the new telemetry batch.

    Responsibilities:
        - Extract a primary numeric column for time-series visualization
        - Compute rolling statistics (mean, std)
        - Compute a simple FFT magnitude spectrum (optional)
        - Detect anomalies based on variance spikes or out-of-range values
        - Return:
            (1) Visualization-ready data for VisualizationTabs
            (2) Optional health summary for HealthSummary

    Returns:
        result: dict
            {
                "x": [...],
                "y": [...],
                "label": "Temperature"
            }

        health: dict | None
            {
                "status": "OK" | "Warning" | "Error",
                "message": "..."
            }
    """

    if df is None or df.empty:
        return {}, None

    # ---------------------------------------------------------
    # 1. Select a numeric column for visualization
    # ---------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {}, None

    col = numeric_cols[0]  # primary metric
    y = df[col].values
    x = np.arange(len(y))

    # ---------------------------------------------------------
    # 2. Compute basic statistics
    # ---------------------------------------------------------
    mean_val = float(np.mean(y))
    std_val = float(np.std(y))

    # ---------------------------------------------------------
    # 3. Optional FFT (very lightweight)
    # ---------------------------------------------------------
    try:
        fft_vals = np.abs(np.fft.rfft(y))
        fft_peak = float(np.max(fft_vals)) if len(fft_vals) > 0 else 0.0
    except Exception:
        fft_peak = 0.0

    # ---------------------------------------------------------
    # 4. Health evaluation
    # ---------------------------------------------------------
    health = None

    # Simple anomaly rules
    if std_val > 5 * (mean_val + 1e-6):
        health = {
            "status": "Warning",
            "message": f"High variance detected in {col} (std={std_val:.2f})"
        }

    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        health = {
            "status": "Error",
            "message": f"Invalid values detected in {col}"
        }

    # ---------------------------------------------------------
    # 5. Visualization output
    # ---------------------------------------------------------
    result = {
        "x": x.tolist(),
        "y": y.tolist(),
        "label": col
    }

    return result, health