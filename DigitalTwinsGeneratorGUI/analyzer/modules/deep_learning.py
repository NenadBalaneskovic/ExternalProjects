# analyzer/modules/deep_learning.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import IsolationForest


def run(df: pd.DataFrame) -> Tuple[Dict[str, Any], Optional[Dict[str, str]]]:
    """
    Lightweight anomaly scoring module inspired by deep-learning autoencoders.

    Rationale:
        True deep learning (autoencoders, CNNs, LSTMs) is too heavy for
        real-time GUI analysis loops. Instead, IsolationForest provides:
            - fast anomaly scoring
            - robust behavior on small batches
            - no GPU requirement
            - similar conceptual output: anomaly score per row

    Responsibilities:
        - Select numeric columns
        - Fit IsolationForest on the current batch
        - Produce anomaly scores (higher = more anomalous)
        - Detect high anomaly ratios and emit health warnings

    Returns:
        result: dict
            {
                "x": [...],
                "anomaly_score": [...],
                "columns": [...]
            }

        health: dict | None
            {
                "status": "Warning" | "Error",
                "message": "..."
            }
    """

    if df is None or df.empty:
        return {}, None

    # ---------------------------------------------------------
    # 1. Select numeric columns
    # ---------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {}, None

    data = df[numeric_cols].values
    n = len(data)

    # Need enough samples for anomaly scoring
    if n < 10:
        return {}, None

    # ---------------------------------------------------------
    # 2. Fit IsolationForest
    # ---------------------------------------------------------
    try:
        model = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42,
        )
        model.fit(data)

        # score_samples returns negative anomaly scores → invert
        scores = -model.score_samples(data)

    except Exception:
        return {}, None

    # ---------------------------------------------------------
    # 3. Health evaluation
    # ---------------------------------------------------------
    health = None

    # Mark top 10% as anomalies
    threshold = np.percentile(scores, 90)
    anomalies = scores >= threshold
    anomaly_ratio = float(np.mean(anomalies))

    if anomaly_ratio > 0.3:
        health = {
            "status": "Warning",
            "message": f"High anomaly ratio detected ({anomaly_ratio:.2f})"
        }

    # ---------------------------------------------------------
    # 4. Visualization output
    # ---------------------------------------------------------
    x = np.arange(n)

    result = {
        "x": x.tolist(),
        "anomaly_score": scores.tolist(),
        "columns": numeric_cols,
    }

    return result, health