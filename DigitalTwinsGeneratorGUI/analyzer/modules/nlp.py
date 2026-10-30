# analyzer/modules/nlp.py

import pandas as pd
from typing import Tuple, Dict, Any, Optional
from collections import Counter
import re


def run(df: pd.DataFrame) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Lightweight NLP module for telemetry logs.

    Responsibilities:
        - Extract text-like columns (e.g., "Error Code", "Message")
        - Perform keyword frequency analysis
        - Detect spikes in error-related terms
        - Produce a readable summary for the NLP tab
        - Emit health warnings when necessary

    Returns:
        summary: str
            Human-readable text summary for VisualizationTabs.update_nlp()

        health: dict | None
            {
                "status": "Warning" | "Error",
                "message": "..."
            }
    """

    if df is None or df.empty:
        return "No text data available.", None

    # ---------------------------------------------------------
    # 1. Identify text columns
    # ---------------------------------------------------------
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_cols:
        return "No text columns found in this batch.", None

    # Combine all text columns into one list of strings
    texts = []
    for col in text_cols:
        texts.extend(df[col].dropna().astype(str).tolist())

    if not texts:
        return "No valid text entries found.", None

    # ---------------------------------------------------------
    # 2. Tokenization (very lightweight)
    # ---------------------------------------------------------
    tokens = []
    for t in texts:
        # Lowercase, remove punctuation, split on whitespace
        t_clean = re.sub(r"[^a-zA-Z0-9]+", " ", t.lower())
        tokens.extend(t_clean.split())

    if not tokens:
        return "Text found, but no valid tokens extracted.", None

    # ---------------------------------------------------------
    # 3. Keyword frequency analysis
    # ---------------------------------------------------------
    freq = Counter(tokens)
    most_common = freq.most_common(10)

    # ---------------------------------------------------------
    # 4. Error keyword detection
    # ---------------------------------------------------------
    error_keywords = {"error", "fail", "fault", "critical", "warning", "exception"}
    error_count = sum(freq.get(k, 0) for k in error_keywords)

    health = None
    if error_count > 5:
        health = {
            "status": "Warning",
            "message": f"High frequency of error-related terms ({error_count})"
        }

    # ---------------------------------------------------------
    # 5. Build summary text
    # ---------------------------------------------------------
    summary_lines = [
        "NLP Summary (Top Keywords):",
        "",
    ]

    for word, count in most_common:
        summary_lines.append(f"  - {word}: {count}")

    if error_count > 0:
        summary_lines.append("")
        summary_lines.append(f"Error-related terms detected: {error_count}")

    summary = "\n".join(summary_lines)

    return summary, health