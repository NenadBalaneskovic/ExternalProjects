# visualization/fem_plotter.py
"""
FEM plotter for Sensor Stress Analyzer.
Generates line plots and heatmaps for FEM stress and heat distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath


def plot_fem_line(stress_map: np.ndarray, heat_map: np.ndarray, filename: str = "fem_plot.png"):
    """
    Plot FEM stress and heat distributions as line plots.
    Args:
        stress_map (np.ndarray): Stress values per vertex.
        heat_map (np.ndarray): Heat values per vertex.
        filename (str): Output filename for the plot.
    """
    fig, ax = plt.subplots()
    ax.plot(stress_map, label="Stress", color="tab:blue", linewidth=2)
    ax.plot(heat_map, label="Heat", color="tab:orange", linewidth=2)
    ax.set_title("FEM Stress & Heat")
    ax.set_xlabel("Corner index", labelpad=10)
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fem_heatmap(vertices: np.ndarray, values: np.ndarray, cmap: str, label: str, filename: str):
    """
    Plot a heatmap over an n-gon polygon using interpolated vertex values.
    Args:
        vertices (np.ndarray): Array of polygon vertices (n, 2).
        values (np.ndarray): Values per vertex (stress or heat).
        cmap (str): Colormap for visualization.
        label (str): Label for colorbar.
        filename (str): Output filename for the heatmap.
    """
    polygon_path = MplPath(vertices)

    pad = 0.15
    xmin, ymin = vertices.min(axis=0) - pad
    xmax, ymax = vertices.max(axis=0) + pad
    res = 400
    gx = np.linspace(xmin, xmax, res)
    gy = np.linspace(ymin, ymax, res)
    XX, YY = np.meshgrid(gx, gy)
    grid_points = np.vstack([XX.ravel(), YY.ravel()]).T

    inside = polygon_path.contains_points(grid_points)
    inside_mask = inside.reshape(XX.shape)

    eps = 1e-9
    dists = np.sqrt(((grid_points[:, None, :] - vertices[None, :, :]) ** 2).sum(axis=2)) + eps

    w = 1.0 / dists
    w_norm = w / (w.sum(axis=1, keepdims=True) + eps)
    interp_vals = (w_norm @ values).reshape(XX.shape)

    interp_vals[~inside_mask] = np.nan

    def robust_min_max(arr, mask):
        valid = arr[mask]
        if valid.size == 0:
            return 0.0, 1.0
        vmin = np.nanpercentile(valid, 2)
        vmax = np.nanpercentile(valid, 98)
        if vmin == vmax:
            vmax = vmin + 1e-6
        return vmin, vmax

    vmin, vmax = robust_min_max(interp_vals, inside_mask)

    fig, ax = plt.subplots()
    im = ax.imshow(
        interp_vals,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    xs, ys = vertices[:, 0], vertices[:, 1]
    ax.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], color="black", linewidth=1.2)
    ax.set_aspect("equal")
    ax.set_title(f"{label} heatmap on n-gon")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fem_heatmaps(vertices: np.ndarray, stress_map: np.ndarray, heat_map: np.ndarray):
    """
    Generate both stress and heat heatmaps for FEM analysis.
    Args:
        vertices (np.ndarray): Polygon vertices.
        stress_map (np.ndarray): Stress values per vertex.
        heat_map (np.ndarray): Heat values per vertex.
    """
    plot_fem_heatmap(vertices, stress_map, cmap="coolwarm", label="Stress", filename="fem_stress_heatmap.png")
    plot_fem_heatmap(vertices, heat_map, cmap="hot", label="Heat", filename="fem_heat_heatmap.png")

    # Save stress heatmap also as fem_heatmap.png for backward compatibility
    plot_fem_heatmap(vertices, stress_map, cmap="coolwarm", label="Stress", filename="fem_heatmap.png")
