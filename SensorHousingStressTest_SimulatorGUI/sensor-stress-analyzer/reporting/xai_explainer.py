# reporting/xai_explainer.py
"""
XAI (Explainable AI) module for Sensor Stress Analyzer.
Generates human-readable summaries of simulation results.
"""

from config import APP_NAME, APP_VERSION


def explain_results(results: dict) -> str:
    """
    Generate a textual explanation of simulation results.
    Includes application metadata for governance-ready reporting.
    """

    header = f"{APP_NAME} v{APP_VERSION} — Quantitative Summary\n\n"

    mode = results.get("mode", "Rod Analysis")
    n = results.get("n", "?")
    force = results.get("force", "?")
    heat = results.get("heat", "?")

    stress_max = results.get("stress_max", None)
    heat_max = results.get("heat_max", None)

    explanation = f"In {mode} mode, the analysis highlights stress and thermal resilience across the sensor housing.\n"
    explanation += f"The polygon had {n} corners, subjected to {force} N force and {heat} °C heat.\n"

    if stress_max is not None:
        explanation += f"The maximum stress reached {stress_max:.2f} units.\n"
    if heat_max is not None:
        explanation += f"The maximum heat intensity was {heat_max:.2f} units.\n"

    explanation += "These values highlight the most critical regions for structural integrity and thermal resilience."

    return header + explanation
