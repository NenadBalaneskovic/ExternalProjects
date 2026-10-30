"""
plots.py

Responsible for:
- Generating memory/time/speedup plots from profiling metrics
- Saving plots to memalloc_data/plots/
- Providing deterministic, reproducible visualizations for the GUI
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Dict, Optional

plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.edgecolor"] = "white"


@dataclass
class PlotPaths:
    memory_plot: Optional[Path]
    runtime_plot: Optional[Path]
    speedup_plot: Optional[Path]


class PlotGenerator:
    def __init__(self, plot_dir: Path):
        self.plot_dir = plot_dir
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def _save_plot(self, out_path: Path):
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, format="png", transparent=False)
        plt.close()

    def generate_plots(self, metrics: List[Dict]) -> PlotPaths:
        if not metrics:
            return PlotPaths(None, None, None)

        # baseline first, optimized second
        metrics = sorted(metrics, key=lambda m: (m["optimized"], m["timestamp"]))
        print("DEBUG METRICS RECEIVED BY PLOT GENERATOR:")
        for m in metrics:
            print(m)

        memory_plot = self.generate_memory_plot(metrics)
        runtime_plot = self.generate_runtime_plot(metrics)
        speedup_plot = self.generate_speedup_plot(metrics)

        return PlotPaths(memory_plot, runtime_plot, speedup_plot)

    def generate_memory_plot(self, metrics: List[Dict]) -> Optional[Path]:
        if not metrics:
            return None

        x = list(range(len(metrics)))
        labels = [m["timestamp"] for m in metrics]
        memory = [m["peak_memory_mb"] for m in metrics]

        plt.figure(figsize=(10, 4))
        plt.plot(x, memory, marker="o", color="blue")
        plt.title("Peak Memory Usage Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Peak Memory (MB)")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.grid(True)

        out_path = self.plot_dir / "memory_usage.png"
        self._save_plot(out_path)
        return out_path

    def generate_runtime_plot(self, metrics: List[Dict]) -> Optional[Path]:
        if not metrics:
            return None

        x = list(range(len(metrics)))
        labels = [m["timestamp"] for m in metrics]
        runtime = [m["runtime_seconds"] for m in metrics]

        plt.figure(figsize=(10, 4))
        plt.plot(x, runtime, marker="o", color="green")
        plt.title("Runtime Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Runtime (seconds)")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.grid(True)

        out_path = self.plot_dir / "runtime.png"
        self._save_plot(out_path)
        return out_path

    def generate_speedup_plot(self, metrics: List[Dict]) -> Optional[Path]:
        if not metrics:
            return None

        x = list(range(len(metrics)))
        labels = [m["timestamp"] for m in metrics]
        speedup = [m.get("speedup", 1.0) for m in metrics]

        plt.figure(figsize=(10, 4))
        plt.plot(x, speedup, marker="o", color="red")
        plt.title("Speedup Over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Speedup")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.grid(True)

        out_path = self.plot_dir / "speedup.png"
        self._save_plot(out_path)
        return out_path

