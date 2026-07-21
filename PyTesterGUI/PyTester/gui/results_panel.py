"""
ResultsPanel

This panel is responsible for:
- displaying collected test execution results
- showing pytest JSON report summaries
- showing coverage summaries
- showing generated plots (durations, failures, coverage)
- providing a final overview of the entire PyTester pipeline

It uses visualization subsystems:
    • PlotResults
    • PlotDurations
    • PlotFailures
    • PlotCoverage
    • PNGExporter

It reads reports from workspace/test_reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QMessageBox
)

# 🔥 KORREKTER IMPORT (CoverageRunner liegt in executor/)
from executor.coverage_runner import CoverageRunner


class ResultsPanel(QWidget):
    """
    GUI panel for displaying test execution results and visualizations.
    """

    def __init__(self, settings: Dict[str, Any], subsystems: Dict[str, Any]) -> None:
        super().__init__()

        self.settings = settings
        self.subsystems = subsystems
        self.summary: Dict[str, Any] = {}
        self.durations: Dict[str, float] = {}
        self.coverage_files: Dict[str, Any] = {}

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title_label = QLabel("No results loaded.")
        layout.addWidget(self.title_label)

        button_row = QHBoxLayout()
        layout.addLayout(button_row)

        self.load_button = QPushButton("Load Results")
        self.load_button.clicked.connect(self.load_results)
        button_row.addWidget(self.load_button)

        self.plot_button = QPushButton("Generate Plots")
        self.plot_button.clicked.connect(self.generate_plots)
        button_row.addWidget(self.plot_button)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(self.output_area)

    # ------------------------------------------------------------
    # Load results
    # ------------------------------------------------------------
    def load_results(self) -> None:
        report_dir = Path(self.settings["paths"]["test_reports"])
        json_report = report_dir / "pytest_report.json"

        if not json_report.exists():
            QMessageBox.warning(self, "No Report", "pytest_report.json not found.")
            return

        import json
        with json_report.open("r", encoding="utf-8") as f:
            pytest_result = json.load(f)

        # ------------------------------------------------------------
        # Coverage separat über CoverageRunner ermitteln
        # ------------------------------------------------------------
        test_files = [Path(p) for p in pytest_result.get("files", [])]
        source_dir = Path(self.settings["paths"]["source"])

        coverage_runner = CoverageRunner(self.settings)
        coverage_result = coverage_runner.run(
            test_files=test_files,
            source_dir=source_dir
        )

        # ------------------------------------------------------------
        # ReportCollector zusammenführen
        # ------------------------------------------------------------
        collector = self.subsystems["report_collector"]
        self.summary = collector.collect(
            pytest_result=pytest_result,
            coverage_result=coverage_result,
            logs=None,
        )

        # ------------------------------------------------------------
        # Plots: echte Durations + echte Coverage-Files
        # ------------------------------------------------------------
        self.durations = pytest_result.get("durations", {
            "pytest": 0.0,
            "coverage": 0.0,
            "total": 0.0,
        })

        self.coverage_files = coverage_result.get("files", {})

        # ------------------------------------------------------------
        # Textausgabe
        # ------------------------------------------------------------
        text_output = self._format_results(self.summary)
        self.output_area.setText(text_output)
        self.title_label.setText("Results Loaded")

    # ------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------
    def generate_plots(self) -> None:
        if not self.summary:
            QMessageBox.warning(self, "No Results", "Load results before generating plots.")
            return

        plot_results = self.subsystems["plot_results"]
        plot_durations = self.subsystems["plot_durations"]
        plot_failures = self.subsystems["plot_failures"]
        plot_coverage = self.subsystems["plot_coverage"]
        exporter = self.subsystems["png_exporter"]

        report = self.summary
        pytest_stdout = report.get("pytest", {}).get("stdout", "")
        durations = self.durations
        coverage_files = self.coverage_files

        results_fig = plot_results.create(report)
        durations_fig = plot_durations.create(durations)
        failures_fig = plot_failures.create(pytest_stdout)
        coverage_fig = plot_coverage.create(coverage_files)

        exporter.export({
            "results_plot": results_fig,
            "durations_plot": durations_fig,
            "failures_plot": failures_fig,
            "coverage_plot": coverage_fig,
        })

        QMessageBox.information(self, "Plots Generated", "Plots saved to workspace/plots.")

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------
    def _format_results(self, summary: Dict[str, Any]) -> str:
        lines = ["=== Final Results Summary ===", ""]

        pytest_summary = summary.get("pytest", {})
        lines.append("Pytest Summary:")
        for key, value in pytest_summary.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")

        coverage_summary = summary.get("coverage", {})
        lines.append("Coverage Summary:")
        for key, value in coverage_summary.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")

        metadata = summary.get("metadata", {})
        if metadata:
            lines.append("Metadata:")
            for key, value in metadata.items():
                lines.append(f"  - {key}: {value}")
            lines.append("")

        return "\n".join(lines)
