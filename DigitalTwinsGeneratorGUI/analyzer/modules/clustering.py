# analyzer/modules/clustering.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


def run(df: pd.DataFrame) -> Tuple[Dict[str, Any], Optional[Dict[str, str]]]:
    """
    Runs lightweight clustering on the new telemetry batch.

    Responsibilities:
        - Select up to 2 numeric columns for 2D visualization
        - Standardize data for stable clustering
        - Run KMeans (fast) or fallback to DBSCAN for anomaly detection
        - Return:
            (1) Visualization-ready scatterplot data
            (2) Optional health summary

    Returns:
        result: dict
            {
                "x": [...],
                "y": [...],
                "labels": [...]
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
    # 1. Select numeric columns
    # ---------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        # Not enough numeric data for clustering
        return {}, None

    # Use first two numeric columns for 2D clustering
    cols = numeric_cols[:2]
    data = df[cols].values

    # ---------------------------------------------------------
    # 2. Standardize data
    # ---------------------------------------------------------
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # ---------------------------------------------------------
    # 3. Run clustering
    # ---------------------------------------------------------
    try:
        # KMeans is fast and stable for real-time telemetry
        kmeans = KMeans(n_clusters=3, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(data_scaled)
        method = "kmeans"
    except Exception:
        # Fallback: DBSCAN for anomaly detection
        db = DBSCAN(eps=0.5, min_samples=5)
        labels = db.fit_predict(data_scaled)
        method = "dbscan"

    # ---------------------------------------------------------
    # 4. Health evaluation
    # ---------------------------------------------------------
    health = None

    # Too many clusters → unstable data
    unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
    if unique_labels > 5:
        health = {
            "status": "Warning",
            "message": f"High cluster count ({unique_labels}) detected"
        }

    # DBSCAN noise points
    if -1 in labels:
        noise_ratio = np.mean(labels == -1)
        if noise_ratio > 0.2:
            health = {
                "status": "Warning",
                "message": f"High noise ratio ({noise_ratio:.2f}) in clustering"
            }

    # ---------------------------------------------------------
    # 5. Visualization output
    # ---------------------------------------------------------
    result = {
        "x": data_scaled[:, 0].tolist(),
        "y": data_scaled[:, 1].tolist(),
        "labels": labels.tolist(),
        "method": method,
        "columns": cols,
    }

    return result, health