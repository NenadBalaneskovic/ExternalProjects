"""
PlotCoverage

This subsystem is responsible for:
- visualizing coverage distribution across files
- generating PNG plots for GUI display
- producing deterministic, side-effect-aware output
- writing plots into workspace/plots/

It does not execute tests; it only visualizes coverage data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from core.utils import ensure_dir


class PlotCoverage:
    """
    Generate coverage plots from unified execution reports.

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
    def create(self, coverage_files: Dict[str, Dict[str, Any]]) -> plt.Figure:
        """
        Create a single combined coverage figure for GUI display.
    
        The GUI expects a matplotlib Figure, not a saved PNG.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
    
        files = list(coverage_files.keys())
        if not files:
            ax.set_title("Coverage by File (%)")
            ax.set_ylabel("Coverage (%)")
            ax.set_ylim(0, 100)
            return fig
    
        # Extract REAL coverage values
        values = []
        for f in files:
            entry = coverage_files.get(f, {})
    
            # Prefer real "coverage" (float percent)
            cov = entry.get("coverage")
    
            # Fallback for older synthetic format
            if cov is None:
                cov = entry.get("coverage_percent", 0.0)
    
            # Final fallback
            if cov is None:
                cov = 0.0
    
            values.append(float(cov))
    
        ax.bar(range(len(files)), values, color="steelblue")
        ax.set_title("Coverage by File (%)")
        ax.set_ylabel("Coverage (%)")
    
        # Dynamic y‑limit: max coverage + padding
        ymax = max(values) if values else 100
        ax.set_ylim(0, max(100, ymax + 5))
    
        ax.set_xticks(range(len(files)))
        ax.set_xticklabels(files, rotation=45, ha="right")
    
        return fig

    # ------------------------------------------------------------
    # File‑saving entrypoint (used by batch exporters)
    # ------------------------------------------------------------
    def plot(self, coverage_files: Dict[str, Dict[str, Any]]) -> Dict[str, Path]:
        """
        Generate all coverage plots and save them as PNGs.
        """
        return {
            "coverage_bar_plot": self._plot_bar(coverage_files),
            "coverage_missing_plot": self._plot_missing(coverage_files),
        }

    # ------------------------------------------------------------
    # Coverage percentage bar chart (PNG)
    # ------------------------------------------------------------
    def _plot_bar(self, coverage_files: Dict[str, Dict[str, Any]]) -> Path:
        files = list(coverage_files.keys())
        if not files:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_title("Coverage by File (%)")
            ax.set_ylabel("Coverage (%)")
            ax.set_ylim(0, 100)
            output_path = self.output_dir / "coverage_bar_plot.png"
            fig.savefig(output_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            return output_path

        values = []
        for f in files:
            entry = coverage_files.get(f, {})
            cov = entry.get("coverage")
            if cov is None:
                cov = entry.get("coverage_percent", 0.0)
            if cov is None:
                cov = 0.0
            values.append(cov)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(files)), values, color="steelblue")
        ax.set_title("Coverage by File (%)")
        ax.set_ylabel("Coverage (%)")
        ax.set_ylim(0, 100)

        ax.set_xticks(range(len(files)))
        ax.set_xticklabels(files, rotation=45, ha="right")

        output_path = self.output_dir / "coverage_bar_plot.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return output_path

    # ------------------------------------------------------------
    # Missing lines horizontal bar chart (PNG)
    # ------------------------------------------------------------
    def _plot_missing(self, coverage_files: Dict[str, Dict[str, Any]]) -> Path:
        files = list(coverage_files.keys())
        if not files:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_title("Missing Lines per File")
            ax.set_xlabel("Count")
            output_path = self.output_dir / "coverage_missing_plot.png"
            fig.savefig(output_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            return output_path

        missing_counts = [len(coverage_files[f].get("missing", [])) for f in files]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(files, missing_counts, color="orange")
        ax.set_title("Missing Lines per File")
        ax.set_xlabel("Count")

        output_path = self.output_dir / "coverage_missing_plot.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return output_path
