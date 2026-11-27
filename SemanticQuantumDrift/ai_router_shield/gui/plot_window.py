# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  gui/plot_window.py
ROLE:    Renders a session's trajectory telemetry: a selectable single-metric
         line chart (H-Score / SQD / RG-Flow / Topo-Drift / Phase Blur, over
         repeated submissions of the SAME prompt) on the left, and the
         translation-drift / trace-covariance phase-space scatter on the
         right - each independently exportable as PNG.
================================================================================
"""

import logging
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QPushButton, QFileDialog, QMessageBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

log = logging.getLogger("PlotWindowEngine")

# Dropdown label -> metrics_log column name
METRIC_OPTIONS = {
    "H-Score": "h_score",
    "SQD-Drift": "sqd_val",
    "RG-Flow": "rg_val",
    "Topo-Drift": "topo_drift",
    "Phase Blur": "trace_covariance_y",
}


class MetricsPlotDialog(QDialog):
    """
    Left panel: a line chart of ONE selected metric (via dropdown) across this
    session's steps - i.e. across repeated submissions of the same prompt.
    Right panel: the translation-drift vs. phase-blur scatter, one point per
    step, colored by step index.
    Both panels have their own "PNG speichern" button.
    """
    def __init__(self, session_title: str, rows: list, parent=None):
        super().__init__(parent)
        self.session_title = session_title
        self.rows = rows
        self.setWindowTitle(f"Metrik-Plot - {session_title}")
        self.resize(950, 480)
        self.setModal(True)

        outer_layout = QVBoxLayout(self)

        if not rows:
            outer_layout.addWidget(QLabel("Für diese Sitzung liegen noch keine Trajektorie-Daten vor."))
            return

        panels_layout = QHBoxLayout()
        outer_layout.addLayout(panels_layout)

        # --- Left column: dropdown-selectable single-metric line chart ---
        left_col = QVBoxLayout()
        self.metric_selector = QComboBox()
        self.metric_selector.addItems(list(METRIC_OPTIONS.keys()))
        self.metric_selector.currentTextChanged.connect(self._redraw_line_chart)
        left_col.addWidget(self.metric_selector)

        self.line_figure = Figure(figsize=(4.6, 4.0), facecolor="#0d1117")
        self.line_canvas = FigureCanvas(self.line_figure)
        left_col.addWidget(self.line_canvas)

        btn_save_line = QPushButton("PNG speichern")
        btn_save_line.clicked.connect(lambda: self._save_png(self.line_figure, self.metric_selector.currentText()))
        left_col.addWidget(btn_save_line)
        panels_layout.addLayout(left_col)

        # --- Right column: fixed phase-space scatter ---
        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Phase-Space Projektion"))

        self.scatter_figure = Figure(figsize=(4.6, 4.0), facecolor="#0d1117")
        self.scatter_canvas = FigureCanvas(self.scatter_figure)
        right_col.addWidget(self.scatter_canvas)

        btn_save_scatter = QPushButton("PNG speichern")
        btn_save_scatter.clicked.connect(lambda: self._save_png(self.scatter_figure, "Phase-Space-Projektion"))
        right_col.addWidget(btn_save_scatter)
        panels_layout.addLayout(right_col)

        self._redraw_line_chart(self.metric_selector.currentText())
        self._draw_scatter()

    # ------------------------------------------------------------- drawing
    def _style_axes(self, ax):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    def _redraw_line_chart(self, metric_label: str):
        column = METRIC_OPTIONS[metric_label]
        steps = [r["step"] for r in self.rows]
        values = [r[column] for r in self.rows]

        self.line_figure.clear()
        ax = self.line_figure.add_subplot(1, 1, 1)
        self._style_axes(ax)

        if metric_label == "H-Score":
            ax.axhspan(0.0, 0.3, color="#238636", alpha=0.15)
            ax.axhspan(0.3, 0.6, color="#d29922", alpha=0.15)
            ax.axhspan(0.6, 1.0, color="#f85149", alpha=0.15)
            ax.set_ylim(0, 1.05)

        ax.plot(steps, values, marker="o", color="#58a6ff", linewidth=2)
        ax.set_xlabel("Step (wiederholte Übermittlung desselben Prompts)", color="#c9d1d9", fontsize=8)
        ax.set_ylabel(metric_label, color="#c9d1d9")
        ax.set_title(f"{metric_label} Verlauf", color="#c9d1d9")
        self.line_figure.tight_layout()
        self.line_canvas.draw()

    def _draw_scatter(self):
        steps = [r["step"] for r in self.rows]
        drift_x = [r["mean_drift_x"] for r in self.rows]
        cov_y = [r["trace_covariance_y"] for r in self.rows]

        self.scatter_figure.clear()
        ax = self.scatter_figure.add_subplot(1, 1, 1)
        self._style_axes(ax)

        scatter = ax.scatter(drift_x, cov_y, c=steps, cmap="viridis", s=60, edgecolors="#c9d1d9")
        ax.set_xlabel("Translation Drift (mean_drift_x)", color="#c9d1d9")
        ax.set_ylabel("Trace Covariance (phase blur)", color="#c9d1d9")
        ax.set_title("Phase-Space Projektion", color="#c9d1d9")
        cbar = self.scatter_figure.colorbar(scatter, ax=ax)
        cbar.set_label("Step", color="#c9d1d9")
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        self.scatter_figure.tight_layout()
        self.scatter_canvas.draw()

    # --------------------------------------------------------------- export
    def _save_png(self, figure: Figure, default_name_hint: str):
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in self.session_title)[:40].strip()
        safe_metric = "".join(c if c.isalnum() or c in "-_ " else "_" for c in default_name_hint)
        default_path = f"{safe_title}_{safe_metric}.png"

        path, _ = QFileDialog.getSaveFileName(self, "Plot als PNG speichern", default_path, "PNG-Bilder (*.png)")
        if not path:
            return
        try:
            figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
        except OSError as e:
            log.error(f"Failed to save plot PNG to {path}: {e}")
            QMessageBox.critical(self, "Speichern fehlgeschlagen", str(e))
            return
        log.info(f"Plot exported to {path}")


log.info("Production gui/plot_window.py component matrix initialized.")