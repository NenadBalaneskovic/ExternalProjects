"""
PlotDurations

This subsystem is responsible for:
- visualizing execution durations (pytest, coverage, total)
- generating PNG plots for GUI display
- producing deterministic, side-effect-aware output
- writing plots into workspace/plots/

It does not execute tests; it only visualizes timing data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from core.utils import ensure_dir


class PlotDurations:
    """
    Generate duration plots from unified execution reports.

    Parameters
    ----------
    settings : dict
        Global configuration loaded from settings.yaml.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        # IMPORTANT:
        # Do NOT create QWidget, FigureCanvas, or any Qt object here.
        # Only store configuration and ensure directories exist.
        self.settings: Dict[str, Any] = settings
        self.output_dir: Path = Path(settings["visualization"]["output_dir"])
        ensure_dir(self.output_dir)

    # ------------------------------------------------------------
    # GUI entrypoint: return a matplotlib Figure
    # ------------------------------------------------------------
    def create(self, durations: Dict[str, float]) -> plt.Figure:
        """
        Create a single combined duration figure for GUI display.
    
        The GUI expects a matplotlib Figure, not a saved PNG.
        """
    
        # Always ensure numeric values
        try:
            pytest_dur = float(durations.get("pytest", 0.0))
        except Exception:
            pytest_dur = 0.0
    
        try:
            coverage_dur = float(durations.get("coverage", 0.0))
        except Exception:
            coverage_dur = 0.0
    
        # If total is missing, compute it
        try:
            total_dur = float(durations.get("total", pytest_dur + coverage_dur))
        except Exception:
            total_dur = pytest_dur + coverage_dur
    
        labels = ["pytest", "coverage", "total"]
        values = [pytest_dur, coverage_dur, total_dur]
    
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, values, color=["blue", "green", "purple"])
        ax.set_title("Execution Durations (seconds)")
        ax.set_ylabel("Seconds")
    
        # Dynamic y-limit: add padding above max value
        ymax = max(values) if values else 1.0
        ax.set_ylim(0, ymax + 0.1 * ymax)
    
        return fig
        
    # ------------------------------------------------------------
    # File‑saving entrypoint (used by batch exporters)
    # ------------------------------------------------------------
    def plot(self, durations: Dict[str, float]) -> Dict[str, Path]:
        """
        Generate all duration plots and save them as PNGs.
        """
        return {
            "duration_bar_plot": self._plot_bar(durations),
            "duration_breakdown_plot": self._plot_breakdown(durations),
        }

    # ------------------------------------------------------------
    # Bar plot (simple overview) — PNG
    # ------------------------------------------------------------
    def _plot_bar(self, durations: Dict[str, float]) -> Path:
        pytest_dur = durations.get("pytest") or 0.0
        coverage_dur = durations.get("coverage") or 0.0
        total_dur = durations.get("total") or (pytest_dur + coverage_dur)

        labels = ["pytest", "coverage", "total"]
        values = [pytest_dur, coverage_dur, total_dur]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, values, color=["blue", "green", "purple"])
        ax.set_title("Execution Durations (seconds)")
        ax.set_ylabel("Seconds")

        output_path = self.output_dir / "duration_bar_plot.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return output_path

    # ------------------------------------------------------------
    # Breakdown plot (pie chart) — PNG
    # ------------------------------------------------------------
    def _plot_breakdown(self, durations: Dict[str, float]) -> Path:
        pytest_dur = durations.get("pytest") or 0.0
        coverage_dur = durations.get("coverage") or 0.0

        labels = []
        values = []

        # FIX: Only include non-zero values, but ensure pie chart is valid
        if pytest_dur > 0:
            labels.append("pytest")
            values.append(pytest_dur)

        if coverage_dur > 0:
            labels.append("coverage")
            values.append(coverage_dur)

        # If everything is zero → fallback
        if not values:
            labels = ["no data"]
            values = [1]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("Execution Duration Breakdown")

        output_path = self.output_dir / "duration_breakdown_plot.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return output_path
