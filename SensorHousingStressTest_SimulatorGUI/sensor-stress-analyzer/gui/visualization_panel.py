# gui/visualization_panel.py

import os
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.path import Path as MplPath

from visualization.rod_plotter import plot_rod_structure
from reporting.xai_explainer import explain_results


class VisualizationPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Rod structure section
        self.rod_label = QLabel("Rod Structure")
        self.rod_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.rod_label)

        self.rod_canvas = FigureCanvas(plt.figure())
        self.rod_canvas.setMinimumHeight(300)
        self.layout.addWidget(self.rod_canvas, stretch=3)

        # FEM Analysis Results section
        self.fem_label = QLabel("FEM Analysis Results")
        self.fem_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.fem_label)

        self.fem_line_canvas = FigureCanvas(plt.figure())
        self.fem_line_canvas.setMinimumHeight(250)
        self.layout.addWidget(self.fem_line_canvas)

        self.heatmaps_row = QWidget()
        self.heatmaps_layout = QHBoxLayout()
        self.heatmaps_row.setLayout(self.heatmaps_layout)
        self.layout.addWidget(self.heatmaps_row)

        self.stress_heatmap_canvas = FigureCanvas(plt.figure())
        self.stress_heatmap_canvas.setMinimumHeight(250)
        self.heat_heatmap_canvas = FigureCanvas(plt.figure())
        self.heat_heatmap_canvas.setMinimumHeight(250)
        self.heatmaps_layout.addWidget(self.stress_heatmap_canvas)
        self.heatmaps_layout.addWidget(self.heat_heatmap_canvas)

        # XAI summary + Log
        self.xai_label = QLabel("Quantitative Summary (XAI)")
        self.xai_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.xai_label)
        self.xai_text = QTextEdit()
        self.xai_text.setReadOnly(True)
        self.layout.addWidget(self.xai_text)

        self.log_label = QLabel("Log window")
        self.log_label.setFont(QFont("Arial", 12))
        self.layout.addWidget(self.log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.layout.addWidget(self.log_text)

        self.save_pdf_button = QPushButton("Save PDF")
        self.save_data_button = QPushButton("Save Data Sets")
        self.layout.addWidget(self.save_pdf_button)
        self.layout.addWidget(self.save_data_button)

    def update_rod_plot(self, n, force, heat):
        self.rod_label.setText(f"Rod Structure: {n} corners, {force} N, {heat} °C")
        data = plot_rod_structure(n, force, heat)

        fig, ax = plt.subplots()
        xs, ys = zip(*data["vertices"])
        xs_poly = list(xs) + [xs[0]]
        ys_poly = list(ys) + [ys[0]]
        ax.plot(xs_poly, ys_poly, 'k-', linewidth=1.5)

        fx, fy = zip(*data["forces"])
        ax.quiver(xs, ys, fx, fy, color='r', angles='xy', scale_units='xy', scale=1)

        ax.set_aspect('equal')
        ax.set_title("Rod Structure")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig("rod_plot.png", dpi=300, bbox_inches="tight")

        self.rod_canvas.figure = fig
        self.rod_canvas.draw()
        self._log_saved("rod_plot.png")

    def update_fem_plot(self, stress_map, heat_map):
        fig, ax = plt.subplots()
        ax.plot(stress_map, label="Stress", color='tab:blue', linewidth=2)
        ax.plot(heat_map, label="Heat", color='tab:orange', linewidth=2)
        ax.set_title("FEM Stress & Heat")
        ax.set_xlabel("Corner index", labelpad=10)
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")
        fig.subplots_adjust(bottom=0.2)
        fig.savefig("fem_plot.png", dpi=300, bbox_inches="tight")

        self.fem_line_canvas.figure = fig
        self.fem_line_canvas.draw()
        self._log_saved("fem_plot.png")
        
    def update_fem_heatmaps(self, n, stress_map, heat_map):
        rod_data = plot_rod_structure(n, force=1, heat=1)
        vertices = np.array(rod_data["vertices"])
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

        stress_vals = np.array(stress_map)
        w_stress = 1.0 / dists
        w_stress_norm = w_stress / (w_stress.sum(axis=1, keepdims=True) + eps)
        stress_interp = (w_stress_norm @ stress_vals).reshape(XX.shape)

        heat_vals = np.array(heat_map)
        w_heat = 1.0 / dists
        w_heat_norm = w_heat / (w_heat.sum(axis=1, keepdims=True) + eps)
        heat_interp = (w_heat_norm @ heat_vals).reshape(XX.shape)

        stress_interp[~inside_mask] = np.nan
        heat_interp[~inside_mask] = np.nan

        def robust_min_max(arr, mask):
            valid = arr[mask]
            if valid.size == 0:
                return 0.0, 1.0
            vmin = np.nanpercentile(valid, 2)
            vmax = np.nanpercentile(valid, 98)
            if vmin == vmax:
                vmax = vmin + 1e-6
            return vmin, vmax

        s_min, s_max = robust_min_max(stress_interp, inside_mask)
        h_min, h_max = robust_min_max(heat_interp, inside_mask)

        # Stress heatmap
        fig_s, ax_s = plt.subplots()
        im_s = ax_s.imshow(
            stress_interp,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="coolwarm",
            vmin=s_min,
            vmax=s_max,
            interpolation="bilinear",
        )
        xs, ys = vertices[:, 0], vertices[:, 1]
        ax_s.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], color="black", linewidth=1.2)
        ax_s.set_aspect("equal")
        ax_s.set_title("Stress heatmap on n-gon")
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        fig_s.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04, label="Stress")
        fig_s.tight_layout()
        fig_s.savefig("fem_stress_heatmap.png", dpi=300, bbox_inches="tight")

        self.stress_heatmap_canvas.figure = fig_s
        self.stress_heatmap_canvas.draw()
        self._log_saved("fem_stress_heatmap.png")

        # Heat heatmap
        fig_h, ax_h = plt.subplots()
        im_h = ax_h.imshow(
            heat_interp,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="hot",
            vmin=h_min,
            vmax=h_max,
            interpolation="bilinear",
        )
        ax_h.plot(list(xs) + [xs[0]], list(ys) + [ys[0]], color="black", linewidth=1.2)
        ax_h.set_aspect("equal")
        ax_h.set_title("Heat heatmap on n-gon")
        ax_h.set_xticks([])
        ax_h.set_yticks([])
        fig_h.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04, label="Heat")
        fig_h.tight_layout()
        fig_h.savefig("fem_heat_heatmap.png", dpi=300, bbox_inches="tight")

        self.heat_heatmap_canvas.figure = fig_h
        self.heat_heatmap_canvas.draw()
        self._log_saved("fem_heat_heatmap.png")

        # Save stress heatmap also as fem_heatmap.png for backward compatibility
        fig_s.savefig("fem_heatmap.png", dpi=300, bbox_inches="tight")
        self._log_saved("fem_heatmap.png")

    def update_xai_summary(self, results):
        explanation = explain_results(results)
        self.xai_text.setText(explanation)

    def log_message(self, message):
        self.log_text.append(message)

    def _log_saved(self, filename):
        if os.path.exists(filename):
            self.log_text.append(f"Saved: {filename}")
