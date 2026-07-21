"""
PlotResults

This subsystem is responsible for:
- visualizing pytest + coverage results
- generating PNG plots for GUI display
- producing deterministic, side-effect-aware output
- writing plots into workspace/plots/

It does not execute tests; it only visualizes results.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from core.utils import ensure_dir


class PlotResults:
    """
    Generate result plots from unified execution reports.

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
    def create(self, report: Dict[str, Any]) -> plt.Figure:
        """
        Create a single combined results figure for GUI display.
    
        The GUI expects a matplotlib Figure, not a saved PNG.
        """
    
        # Extract global execution status (pytest + coverage merged)
        status_str = report.get("status", "unknown")
    
        # Convert status to numeric value for plotting
        status_val = 1.0 if status_str == "ok" else 0.0
    
        # Extract real total coverage (float percent)
        total_cov = report.get("summary", {}).get("total_coverage")
        if total_cov is None:
            total_cov = 0.0
    
        # Ensure numeric
        try:
            total_cov = float(total_cov)
        except Exception:
            total_cov = 0.0
    
        fig, ax = plt.subplots(figsize=(6, 4))
    
        ax.bar(
            ["status", "coverage"],
            [status_val, total_cov],
            color=[
                "green" if status_val == 1.0 else "red",
                "blue"
            ]
        )
    
        ax.set_title("Execution Results")
        ax.set_ylabel("Value")
    
        # Dynamic y-limit: coverage may be < 100
        ymax = max(status_val, total_cov)
        ax.set_ylim(0, ymax + 0.1 * ymax + 1)
    
        return fig

    # ------------------------------------------------------------
    # File‑saving entrypoint (used by batch exporters)
    # ------------------------------------------------------------
    def plot(self, report: Dict[str, Any]) -> Dict[str, Path]:
        """
        Generate all result plots and save them as PNGs.
        """
        return {
            "status_plot": self._plot_status(report),
            "coverage_plot": self._plot_coverage(report),
            "missing_lines_plot": self._plot_missing_lines(report),
        }

    # ------------------------------------------------------------
    # Status plot (PNG)
    # ------------------------------------------------------------
    def _plot_status(self, report: Dict[str, Any]) -> Path:
        status = report.get("status", "unknown")

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(["status"], [1], color="green" if status == "ok" else "red")
        ax.set_title(f"Execution Status: {status}")
        ax.set_ylim(0, 1.2)

        output_path = self.output_dir / "status_plot.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return output_path

    # ------------------------------------------------------------
    # Coverage plot (PNG)
    # ------------------------------------------------------------
    def _plot_coverage(self, report: Dict[str, Any]) -> Path:
        total = report.get("summary", {}).get("total_coverage")
        if total is None:
            total = 0.0

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["coverage"], [total], color="blue")
        ax.set_title("Total Coverage (%)")
        ax.set_ylim(0, 100)

        output_path = self.output_dir / "coverage_plot.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return output_path

    # ------------------------------------------------------------
    # Missing lines plot (PNG)
    # ------------------------------------------------------------
    def _plot_missing_lines(self, report: Dict[str, Any]) -> Path:
        missing = report.get("summary", {}).get("missing_lines", {})

        files = list(missing.keys())
        counts = [len(missing[f]) for f in files]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(files, counts, color="orange")
        ax.set_title("Missing Lines per File")
        ax.set_ylabel("Count")

        # FIX: ensure labels render correctly
        ax.set_xticks(range(len(files)))
        ax.set_xticklabels(files, rotation=45, ha="right")

        output_path = self.output_dir / "missing_lines_plot.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return output_path
