# visualization/color_maps.py
"""
Centralized colormap definitions for Sensor Stress Analyzer.
Ensures consistent visual styling across rod and FEM plots.
"""

import matplotlib.pyplot as plt


# --- Stress colormap ---
def get_stress_colormap():
    """
    Return the colormap used for stress visualization.
    """
    return plt.get_cmap("coolwarm")


# --- Heat colormap ---
def get_heat_colormap():
    """
    Return the colormap used for heat visualization.
    """
    return plt.get_cmap("hot")


# --- Utility for governance-ready consistency ---
def get_colormap(label: str):
    """
    Return the appropriate colormap based on label.
    Args:
        label (str): Either 'stress' or 'heat'.
    Returns:
        matplotlib.colors.Colormap
    """
    label = label.lower()
    if label == "stress":
        return get_stress_colormap()
    elif label == "heat":
        return get_heat_colormap()
    else:
        raise ValueError(f"Unsupported colormap label: {label}")
