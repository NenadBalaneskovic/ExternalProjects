# simulation/rod_analysis.py
"""
Rod analysis module for Sensor Stress Analyzer.
Performs simplified stress and heat calculations for polygonal rod structures.
"""

import numpy as np


def run_rod_analysis(n: int, force: float, heat: float) -> dict:
    """
    Perform a simplified rod analysis on an n-gon structure.
    Returns stress and heat distributions along with maxima for reporting.
    """

    # --- Geometry setup ---
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vertices = np.column_stack((np.cos(angles), np.sin(angles)))

    # --- Simplified stress model ---
    # Stress proportional to applied force and vertex angle
    stress_map = force * (1 + 0.1 * np.sin(angles))
    stress_max = float(np.max(stress_map))

    # --- Simplified heat model ---
    # Heat proportional to applied temperature and vertex angle
    heat_map = heat * (1 + 0.05 * np.cos(angles))
    heat_max = float(np.max(heat_map))

    # --- Results dictionary ---
    results = {
        "mode": "Rod Analysis",
        "n": n,
        "force": force,
        "heat": heat,
        "vertices": vertices.tolist(),
        "stress_map": stress_map.tolist(),
        "heat_map": heat_map.tolist(),
        "stress_max": stress_max,
        "heat_max": heat_max,
    }

    return results
