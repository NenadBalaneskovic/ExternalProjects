"""
PlotFailures

This subsystem is responsible for:
- visualizing test failures extracted from execution reports
- generating PNG plots for GUI display
- producing deterministic, side-effect-aware output
- writing plots into workspace/plots/

It does not execute tests; it only visualizes failure data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from core.utils import ensure_dir


class PlotFailures:
    """
    Generate failure plots from unified execution reports.

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
    def create(self, pytest_stdout: str) -> plt.Figure:
        """
        Create a single combined failure figure for GUI display.
    
        The GUI expects a matplotlib Figure, not a saved PNG.
        """
    
        # Extract real pass/fail counts
        passed, failed, _ = self._extract_failures(pytest_stdout)
    
        # Ensure numeric values
        try:
            passed = int(passed)
        except Exception:
            passed = 0
    
        try:
            failed = int(failed)
        except Exception:
            failed = 0
    
        labels = ["passed", "failed"]
        values = [passed, failed]
    
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, values, color=["green", "red"])
        ax.set_title("Test Results")
        ax.set_ylabel("Count")
    
        # Dynamic y-limit: add padding above max value
        ymax = max(values) if values else 1
        ax.set_ylim(0, ymax + 0.1 * ymax + 1)
    
        return fig

    # ------------------------------------------------------------
    # File‑saving entrypoint (used by batch exporters)
    # ------------------------------------------------------------
    def plot(self, pytest_stdout: str) -> Dict[str, Path]:
        """
        Generate all failure plots and save them as PNGs.
        """
        passed, failed, per_file = self._extract_failures(pytest_stdout)

        passed = passed or 0
        failed = failed or 0

        return {
            "failure_bar_plot": self._plot_bar(passed, failed),
            "failure_pie_plot": self._plot_pie(passed, failed),
        }

    # ------------------------------------------------------------
    # Failure extraction
    # ------------------------------------------------------------
    def _extract_failures(self, stdout: str) -> (int, int, Dict[str, int]):
        """
        Extract pass/fail counts and per-file failures from pytest stdout.

        Returns
        -------
        (passed, failed, per_file)
        """
        passed = 0
        failed = 0
        per_file: Dict[str, int] = {}

        if not stdout:
            return passed, failed, per_file

        for line in stdout.splitlines():
            line = line.strip()

            # Example: "3 passed, 1 failed"
            if "passed" in line or "failed" in line:
                parts = line.replace(",", "").split()
                for i, p in enumerate(parts):
                    if p == "passed":
                        try:
                            passed = int(parts[i - 1])
                        except Exception:
                            pass
                    if p == "failed":
                        try:
                            failed = int(parts[i - 1])
                        except Exception:
                            pass

            # Example: "FAILED test_math.py::test_addition"
            if line.startswith("FAILED"):
                try:
                    file_part = line.split()[1]
                    file_name = file_part.split("::")[0]
                    per_file[file_name] = per_file.get(file_name, 0) + 1
                except Exception:
                    continue

        return passed, failed, per_file

    # ------------------------------------------------------------
    # Bar plot (PNG)
    # ------------------------------------------------------------
    def _plot_bar(self, passed: int, failed: int) -> Path:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["passed", "failed"], [passed, failed], color=["green", "red"])
        ax.set_title("Test Results")
        ax.set_ylabel("Count")

        output_path = self.output_dir / "failure_bar_plot.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return output_path

    # ------------------------------------------------------------
    # Pie chart (PNG)
    # ------------------------------------------------------------
    def _plot_pie(self, passed: int, failed: int) -> Path:
        labels = []
        values = []

        # FIX: Only include non-zero values, but ensure pie chart is valid
        if passed > 0:
            labels.append("passed")
            values.append(passed)

        if failed > 0:
            labels.append("failed")
            values.append(failed)

        # If everything is zero → fallback
        if not values:
            labels = ["no data"]
            values = [1]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("Pass/Fail Ratio")

        output_path = self.output_dir / "failure_pie_plot.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return output_path
