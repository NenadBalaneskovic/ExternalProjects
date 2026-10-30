"""
PNGExporter

This subsystem is responsible for:
- exporting matplotlib figures or raw PNG paths into workspace/plots/
- providing a unified interface for GUI components
- ensuring deterministic, side-effect-aware file output
- normalizing filenames and paths

It does not generate plots; it only exports them.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Any, Union

from core.utils import ensure_dir


class PNGExporter:
    """
    Export PNG files into workspace/plots/ for GUI consumption.

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
    # Main entrypoint
    # ------------------------------------------------------------
    def export(self, items: Dict[str, Union[Path, "plt.Figure"]]) -> Dict[str, Path]:
        """
        Export multiple PNG items into workspace/plots/.

        Each item may be:
        - a Path to an existing PNG
        - a matplotlib Figure object

        Parameters
        ----------
        items : dict
            {
                "coverage_plot": Path or Figure,
                "duration_plot": Path or Figure,
                ...
            }

        Returns
        -------
        dict
            {
                "coverage_plot": Path,
                "duration_plot": Path,
                ...
            }
        """
        exported: Dict[str, Path] = {}

        for name, item in items.items():
            dst_path = self._export_single(name, item)
            exported[name] = dst_path

        return exported

    # ------------------------------------------------------------
    # Single-item export
    # ------------------------------------------------------------
    def _export_single(self, name: str, item: Union[Path, "plt.Figure"]) -> Path:
        """
        Export a single PNG item.

        Parameters
        ----------
        name : str
            Logical name of the plot (e.g., "coverage_plot").

        item : Path or Figure
            Either a PNG file path or a matplotlib Figure.

        Returns
        -------
        Path
            Destination PNG file path.
        """
        safe_name = self._normalize_name(name)
        dst_path = self.output_dir / f"{safe_name}.png"

        # Case 1: matplotlib Figure → save directly
        try:
            import matplotlib.pyplot as plt
            if hasattr(item, "savefig"):
                item.savefig(dst_path, dpi=120, bbox_inches="tight")
                plt.close(item)
                return dst_path
        except Exception:
            pass

        # Case 2: raw PNG path → copy
        try:
            shutil.copyfile(item, dst_path)
            return dst_path
        except Exception:
            return self._placeholder_png(dst_path)

    # ------------------------------------------------------------
    # Filename normalization
    # ------------------------------------------------------------
    def _normalize_name(self, name: str) -> str:
        """
        Normalize plot names into safe filenames.

        Example:
        "coverage_plot" → "coverage_plot"
        "duration breakdown" → "duration_breakdown"
        """
        name = name.lower().strip()
        name = name.replace(" ", "_")
        name = name.replace("-", "_")
        return name

    # ------------------------------------------------------------
    # Placeholder PNG creation
    # ------------------------------------------------------------
    def _placeholder_png(self, dst_path: Path) -> Path:
        """
        Create a minimal placeholder PNG if export fails.

        Returns
        -------
        Path
            Path to placeholder PNG.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3, 2))
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.set_axis_off()

        fig.savefig(dst_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return dst_path

    # ------------------------------------------------------------
    # Summary helper
    # ------------------------------------------------------------
    def summarize(self, exported: Dict[str, Path]) -> str:
        """
        Produce a human-readable summary of exported PNGs.

        Parameters
        ----------
        exported : dict
            Mapping of logical plot names to exported file paths.

        Returns
        -------
        str
            Formatted summary.
        """
        lines = ["=== PNG Export Summary ===", ""]
        for name, path in exported.items():
            lines.append(f"{name}: {path}")
        return "\n".join(lines)
