# analyzer/utils/plot_helpers.py

import numpy as np
from typing import Dict, Any, List, Optional


class PlotHelpers:
    """
    Utility class for preparing visualization-friendly data structures.

    Responsibilities:
        - Normalize time-series data
        - Validate scatterplot inputs
        - Prepare anomaly score curves
        - Provide safe defaults for missing or malformed data

    Methods:
        prepare_time_series(result_dict)
        prepare_scatter(result_dict)
        prepare_forecast(result_dict)
        prepare_anomaly_curve(result_dict)
    """

    # ---------------------------------------------------------
    # Time-Series
    # ---------------------------------------------------------
    @staticmethod
    def prepare_time_series(data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Ensures time-series data is well-formed.

        Expected:
            {
                "x": [...],
                "y": [...],
                "label": "Temperature"
            }
        """
        if not data:
            return {"x": [], "y": [], "label": "No Data"}

        x = data.get("x", [])
        y = data.get("y", [])
        label = data.get("label", "Signal")

        # Convert to lists of floats
        try:
            x = [float(v) for v in x]
            y = [float(v) for v in y]
        except Exception:
            x, y = [], []

        return {"x": x, "y": y, "label": label}

    # ---------------------------------------------------------
    # Scatter (Clustering)
    # ---------------------------------------------------------
    @staticmethod
    def prepare_scatter(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures clustering scatterplot data is well-formed.

        Expected:
            {
                "x": [...],
                "y": [...],
                "labels": [...],
                "columns": [...],
                "method": "kmeans"
            }
        """
        if not data:
            return {"x": [], "y": [], "labels": [], "columns": [], "method": "none"}

        try:
            x = [float(v) for v in data.get("x", [])]
            y = [float(v) for v in data.get("y", [])]
            labels = data.get("labels", [])
        except Exception:
            x, y, labels = [], [], []

        return {
            "x": x,
            "y": y,
            "labels": labels,
            "columns": data.get("columns", []),
            "method": data.get("method", "unknown"),
        }

    # ---------------------------------------------------------
    # Forecasting
    # ---------------------------------------------------------
    @staticmethod
    def prepare_forecast(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures forecast data is well-formed.

        Expected:
            {
                "history_x": [...],
                "history_y": [...],
                "forecast_x": [...],
                "forecast_y": [...],
                "label": "Temperature"
            }
        """
        if not data:
            return {
                "history_x": [],
                "history_y": [],
                "forecast_x": [],
                "forecast_y": [],
                "label": "No Data",
            }

        try:
            hx = [float(v) for v in data.get("history_x", [])]
            hy = [float(v) for v in data.get("history_y", [])]
            fx = [float(v) for v in data.get("forecast_x", [])]
            fy = [float(v) for v in data.get("forecast_y", [])]
        except Exception:
            hx, hy, fx, fy = [], [], [], []

        return {
            "history_x": hx,
            "history_y": hy,
            "forecast_x": fx,
            "forecast_y": fy,
            "label": data.get("label", "Signal"),
        }

    # ---------------------------------------------------------
    # Anomaly Curve (Deep Learning)
    # ---------------------------------------------------------
    @staticmethod
    def prepare_anomaly_curve(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures anomaly score data is well-formed.

        Expected:
            {
                "x": [...],
                "anomaly_score": [...],
                "columns": [...]
            }
        """
        if not data:
            return {"x": [], "anomaly_score": [], "columns": []}

        try:
            x = [float(v) for v in data.get("x", [])]
            scores = [float(v) for v in data.get("anomaly_score", [])]
        except Exception:
            x, scores = [], []

        return {
            "x": x,
            "anomaly_score": scores,
            "columns": data.get("columns", []),
        }