# visualization/rod_plotter.py
"""
Rod plotter for Sensor Stress Analyzer.
Generates rod structure plots with applied forces for polygonal rod analysis.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_rod_structure(n: int, force: float, heat: float, filename: str = "rod_plot.png") -> dict:
    """
    Plot the rod structure of an n-gon with applied forces.
    Args:
        n (int): Number of polygon corners.
        force (float): Applied force [N].
        heat (float): Applied heat [°C].
        filename (str): Output filename for the plot.
    Returns:
        dict: Data including vertices and forces for downstream use.
    """

    # --- Geometry setup ---
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    vertices = np.column_stack((np.cos(angles), np.sin(angles)))

    # --- Simplified force vectors ---
    fx = force * np.cos(angles)
    fy = force * np.sin(angles)
    forces = np.column_stack((fx, fy))

    # --- Plot rod structure ---
    fig, ax = plt.subplots()
    xs, ys = vertices[:, 0], vertices[:, 1]
    xs_poly = list(xs) + [xs[0]]
    ys_poly = list(ys) + [ys[0]]
    ax.plot(xs_poly, ys_poly, "k-", linewidth=1.5)

    ax.quiver(xs, ys, fx, fy, color="r", angles="xy", scale_units="xy", scale=1)

    ax.set_aspect("equal")
    ax.set_title(f"Rod Structure: {n} corners, {force} N, {heat} °C")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Return data for downstream use ---
    return {
        "n": n,
        "force": force,
        "heat": heat,
        "vertices": vertices.tolist(),
        "forces": forces.tolist(),
    }
