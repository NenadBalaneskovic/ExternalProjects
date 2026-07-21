"""
ExecutionPanel

This panel is responsible for:
- running pytest on the generated test suite
- capturing logs and execution output
- running coverage analysis
- collecting reports
- displaying execution results to the user

It uses:
    • PytestRunner
    • CoverageRunner
    • ReportCollector
    • LogCapture

It reads test files from workspace/generated_tests.

Minimal erweitert für:
- Multi‑Select (getOpenFileNames)
- Schleife über mehrere Testdateien
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QMessageBox,
    QFileDialog
)


class ExecutionPanel(QWidget):
    """
    GUI panel for executing pytest tests and displaying results.
    """

    def __init__(self, settings: Dict[str, Any], subsystems: Dict[str, Any]) -> None:
        super().__init__()

        self.settings = settings
        self.subsystems = subsystems

        # Multi‑select: now a list
        self.selected_test_files: List[Path] = []

        # ------------------------------------------------------------
        # GUI Layout
        # ------------------------------------------------------------
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.file_label = QLabel("No test files selected.")
        layout.addWidget(self.file_label)

        button_row = QHBoxLayout()
        layout.addLayout(button_row)

        self.select_button = QPushButton("Select Test Files")
        self.select_button.clicked.connect(self.select_test_files)
        button_row.addWidget(self.select_button)

        self.run_button = QPushButton("Run Tests")
        self.run_button.clicked.connect(self.run_tests)
        button_row.addWidget(self.run_button)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(self.output_area)

    # ------------------------------------------------------------
    # Multi‑Select File selection
    # ------------------------------------------------------------
    def select_test_files(self) -> None:
        test_dir = Path(self.settings["paths"]["generated_tests"])

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Generated Test Files",
            str(test_dir),
            "Python Files (*.py)"
        )

        if file_paths:
            self.selected_test_files = [Path(p) for p in file_paths]
            names = ", ".join([p.name for p in self.selected_test_files])
            self.file_label.setText(f"Selected: {names}")

    # ------------------------------------------------------------
    # Test execution pipeline (multi‑file)
    # ------------------------------------------------------------
    def run_tests(self) -> None:
        if not self.selected_test_files:
            QMessageBox.warning(self, "No Test Files", "Please select at least one test file first.")
            return

        # Start Python log capture
        log_capture = self.subsystems["log_capture"]
        log_capture.start_python_capture()

        # Run pytest on all selected test files
        pytest_runner = self.subsystems["pytest_runner"]
        pytest_result = pytest_runner.run(self.selected_test_files)

        # Run coverage (source_dir is fixed)
        coverage_runner = self.subsystems["coverage_runner"]
        coverage_result = coverage_runner.run(
            test_files=self.selected_test_files,
            source_dir=Path(self.settings["paths"]["source"])
        )

        # Stop Python log capture
        python_logs = log_capture.stop_python_capture()

        # Merge logs
        subprocess_logs = log_capture.capture_subprocess_logs(
            pytest_result.get("stdout"),
            pytest_result.get("stderr")
        )
        unified_logs = log_capture.merge(subprocess_logs, python_logs)

        # Collect reports
        collector = self.subsystems["report_collector"]
        report_summary = collector.collect(
            pytest_result,
            coverage_result,
            unified_logs
        )

        # Display results
        text_output = self._format_results(
            pytest_result,
            coverage_result,
            report_summary,
            unified_logs
        )
        self.output_area.setText(text_output)

        QMessageBox.information(self, "Execution Complete", "Tests executed successfully.")

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------
    def _format_results(
        self,
        pytest_result: Dict[str, Any],
        coverage_result: Dict[str, Any],
        report_summary: Dict[str, Any],
        logs: str
    ) -> str:

        lines: list[str] = []
        lines.append("=== Test Execution Results ===\n")

        # Pytest results
        lines.append("Pytest:")
        for key, value in pytest_result.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")

        # Coverage results
        lines.append("Coverage:")
        for key, value in coverage_result.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")

        # Report summary
        lines.append("Reports:")
        for key, value in report_summary.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")

        # Logs
        lines.append("Logs:")
        lines.append(logs)
        lines.append("")

        return "\n".join(lines)
