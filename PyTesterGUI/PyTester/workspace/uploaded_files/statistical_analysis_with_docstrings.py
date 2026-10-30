"""
statistical_analysis.py

A fully documented and test‑friendly Python module for statistical
analysis of a fictitious time‑series dataset.

This file is intentionally designed as an *ideal test case* for the
PyTester GUI. It contains:

- a clean, well‑annotated class
- abundant docstrings
- deterministic structure
- clear separation of concerns
- plotting and CSV export functionality

The class performs:
- mean, median, standard deviation
- correlation between sensors
- autocorrelation of sensor_A
- PNG plot generation
- CSV export of summary statistics
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any


class StatisticalAnalyzer:
    """
    A class for performing statistical analysis on a time‑series dataset.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file containing the fictitious measurement data.

    Attributes
    ----------
    data : pd.DataFrame
        Loaded dataset containing timestamps, sensor values, noise, and event flags.

    stats : Dict[str, float]
        Dictionary storing computed summary statistics.

    output_dir : Path
        Directory where plots and exported CSV files will be saved.
    """

    def __init__(self, csv_path: Path, output_dir: Path = Path("analysis_output")) -> None:
        self.csv_path: Path = csv_path
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(exist_ok=True)

        self.data: pd.DataFrame = pd.read_csv(csv_path)
        self.stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Statistical Computations
    # ------------------------------------------------------------------

    def compute_basic_statistics(self) -> Dict[str, float]:
        """
        Compute mean, median, and standard deviation for sensor_A and sensor_B.

        Returns
        -------
        Dict[str, float]
            A dictionary containing computed statistics.
        """
        sensor_A = self.data["sensor_A"]
        sensor_B = self.data["sensor_B"]

        self.stats = {
            "mean_A": float(sensor_A.mean()),
            "median_A": float(sensor_A.median()),
            "std_A": float(sensor_A.std()),
            "mean_B": float(sensor_B.mean()),
            "median_B": float(sensor_B.median()),
            "std_B": float(sensor_B.std()),
        }

        return self.stats

    def compute_correlation(self) -> float:
        """
        Compute Pearson correlation between sensor_A and sensor_B.

        Returns
        -------
        float
            Correlation coefficient.
        """
        corr = float(self.data["sensor_A"].corr(self.data["sensor_B"]))
        self.stats["correlation_A_B"] = corr
        return corr

    def compute_autocorrelation(self, lag: int = 1) -> float:
        """
        Compute autocorrelation of sensor_A for a given lag.

        Parameters
        ----------
        lag : int
            Time lag for autocorrelation.

        Returns
        -------
        float
            Autocorrelation value.
        """
        sensor_A = self.data["sensor_A"]
        autocorr = float(sensor_A.autocorr(lag=lag))
        self.stats[f"autocorr_A_lag_{lag}"] = autocorr
        return autocorr

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_time_series(self) -> Path:
        """
        Plot sensor_A and sensor_B over time and save as PNG.

        Returns
        -------
        Path
            Path to the saved PNG file.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.data["timestamp"], self.data["sensor_A"], label="Sensor A")
        plt.plot(self.data["timestamp"], self.data["sensor_B"], label="Sensor B")
        plt.xlabel("Timestamp")
        plt.ylabel("Sensor Values")
        plt.title("Time Series of Sensor A and B")
        plt.legend()
        plt.tight_layout()

        out_path = self.output_dir / "time_series.png"
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_correlation(self) -> Path:
        """
        Plot scatter correlation between sensor_A and sensor_B.

        Returns
        -------
        Path
            Path to the saved PNG file.
        """
        plt.figure(figsize=(6, 6))
        plt.scatter(self.data["sensor_A"], self.data["sensor_B"], alpha=0.6)
        plt.xlabel("Sensor A")
        plt.ylabel("Sensor B")
        plt.title("Correlation: Sensor A vs Sensor B")
        plt.tight_layout()

        out_path = self.output_dir / "correlation.png"
        plt.savefig(out_path)
        plt.close()
        return out_path

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_statistics(self) -> Path:
        """
        Export computed statistics to a CSV file.

        Returns
        -------
        Path
            Path to the saved CSV file.
        """
        stats_df = pd.DataFrame([self.stats])
        out_path = self.output_dir / "summary_statistics.csv"
        stats_df.to_csv(out_path, index=False)
        return out_path

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def run_full_analysis(self) -> None:
        """
        Execute the full analysis pipeline:
        - compute statistics
        - compute correlation
        - compute autocorrelation
        - generate plots
        - export summary statistics
        """
        self.compute_basic_statistics()
        self.compute_correlation()
        self.compute_autocorrelation(lag=1)

        self.plot_time_series()
        self.plot_correlation()

        self.export_statistics()


# ----------------------------------------------------------------------
# Example usage (this block is ignored by PyTester but useful for humans)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    analyzer = StatisticalAnalyzer(
        csv_path=Path("fictitious_measurements.csv"),
        output_dir=Path("analysis_output")
    )
    analyzer.run_full_analysis()
    print("Analysis complete.")