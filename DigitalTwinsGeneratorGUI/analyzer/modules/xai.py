# analyzer/modules/xai.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor


def run(df: pd.DataFrame) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Lightweight XAI-style feature attribution for telemetry.

    Rationale:
        Full SHAP/Captum pipelines are too heavy for a real-time GUI loop.
        Instead, we approximate feature importance using a small
        RandomForestRegressor and its built-in feature_importances_.

    Responsibilities:
        - Select numeric columns
        - Choose a primary target signal (first numeric column)
        - Train a tiny RandomForest to predict the target from the others
        - Use feature_importances_ as a proxy for attribution
        - Produce a human-readable explanation summary
        - Emit health warnings if the model is unstable or attribution is degenerate

    Returns:
        summary: str
            Human-readable explanation text for VisualizationTabs.update_xai()

        health: dict | None
            {
                "status": "Warning" | "Error",
                "message": "..."
            }
    """

    if df is None or df.empty:
        return "No data available for explainability.", None

    # ---------------------------------------------------------
    # 1. Select numeric columns
    # ---------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return "Not enough numeric features for XAI analysis.", None

    # Target: first numeric column
    target_col = numeric_cols[0]
    feature_cols = numeric_cols[1:]

    y = df[target_col].values
    X = df[feature_cols].values

    # Need enough samples
    if len(X) < 20:
        return "Too few samples for stable feature attribution.", None

    # ---------------------------------------------------------
    # 2. Train small RandomForest
    # ---------------------------------------------------------
    try:
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=1,
        )
        model.fit(X, y)
        importances = model.feature_importances_
    except Exception:
        return "XAI model training failed on this batch.", None

    # ---------------------------------------------------------
    # 3. Normalize and sort importances
    # ---------------------------------------------------------
    total_importance = float(np.sum(importances))
    if total_importance <= 0:
        return "Feature importances are degenerate (all zero).", {
            "status": "Warning",
            "message": "XAI importances are all zero; model may be unstable."
        }

    normalized = importances / total_importance
    pairs = list(zip(feature_cols, normalized))
    pairs.sort(key=lambda x: x[1], reverse=True)

    # ---------------------------------------------------------
    # 4. Build explanation summary
    # ---------------------------------------------------------
    lines = [
        f"XAI Summary for target: {target_col}",
        "",
        "Relative feature importances:",
        "",
    ]

    for name, imp in pairs:
        lines.append(f"  - {name}: {imp:.3f}")

    # Simple health heuristic: if one feature dominates > 0.9
    health = None
    if pairs[0][1] > 0.9:
        health = {
            "status": "Warning",
            "message": f"XAI: {pairs[0][0]} dominates attribution ({pairs[0][1]:.2f})."
        }

    summary = "\n".join(lines)
    return summary, health