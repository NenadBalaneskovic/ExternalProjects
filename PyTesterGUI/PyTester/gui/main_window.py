"""
Main application window for the PyTester GUI.

This module defines the top-level GUI container that orchestrates
all functional panels:

- UploadPanel
- InspectionPanel
- InferencePanel
- TestGenerationPanel
- ExecutionPanel
- ResultsPanel

It receives:
- global settings (parsed from settings.yaml)
- subsystem instances (initialized in run.py)

The MainWindow class is responsible for:
- constructing the main Qt window
- building the tabbed interface
- wiring each panel with settings + subsystems

The QApplication is created in run.py, not here.
"""

from __future__ import annotations

from typing import Dict, Any

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QTabWidget,
)

# GUI panels
from gui.upload_panel import UploadPanel
from gui.inspection_panel import InspectionPanel
from gui.inference_panel import InferencePanel
from gui.test_generation_panel import TestGenerationPanel
from gui.execution_panel import ExecutionPanel
from gui.results_panel import ResultsPanel


class MainWindow(QMainWindow):
    """
    The main GUI window for PyTester.

    Parameters
    ----------
    settings : dict
        Global configuration loaded from settings.yaml.

    subsystems : dict
        Dictionary containing initialized backend components
        (core, inference, testgen, executor, visualization).
    """

    def __init__(self, settings: Dict[str, Any], subsystems: Dict[str, Any]) -> None:
        super().__init__()

        self.settings: Dict[str, Any] = settings
        self.subsystems: Dict[str, Any] = subsystems

        # ------------------------------------------------------------
        # Window configuration
        # ------------------------------------------------------------
        self.setWindowTitle(self.settings["app"]["name"])
        self.resize(
            self.settings["gui"]["window"]["width"],
            self.settings["gui"]["window"]["height"],
        )

        # ------------------------------------------------------------
        # Central widget + layout
        # ------------------------------------------------------------
        central_widget: QWidget = QWidget()
        main_layout: QVBoxLayout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

                # ------------------------------------------------------------
        # Tab widget (holds all functional panels)
        # ------------------------------------------------------------
        tabs: QTabWidget = QTabWidget()
        main_layout.addWidget(tabs)

        # ------------------------------------------------------------
        # Instantiate all GUI panels
        # ------------------------------------------------------------
        self.upload_panel: UploadPanel = UploadPanel(self.settings, self.subsystems)
        self.inspection_panel: InspectionPanel = InspectionPanel(self.settings, self.subsystems)
        self.inference_panel: InferencePanel = InferencePanel(self.settings, self.subsystems)
        self.test_generation_panel: TestGenerationPanel = TestGenerationPanel(self.settings, self.subsystems)
        self.execution_panel: ExecutionPanel = ExecutionPanel(self.settings, self.subsystems)
        self.results_panel: ResultsPanel = ResultsPanel(self.settings, self.subsystems)

        # ------------------------------------------------------------
        # Add panels to tab widget
        # ------------------------------------------------------------
        tabs.addTab(self.upload_panel, "Upload")
        tabs.addTab(self.inspection_panel, "Inspection")
        tabs.addTab(self.inference_panel, "Inference")
        tabs.addTab(self.test_generation_panel, "Test Generation")
        tabs.addTab(self.execution_panel, "Execution")
        tabs.addTab(self.results_panel, "Results")

    # ------------------------------------------------------------
    # NOTE: QApplication is created in run.py, not here.
    # ------------------------------------------------------------
    # The old .run() method has been removed because it created
    # a second QApplication instance, which caused:
    #
    #   QWidget: Must construct a QApplication before a QWidget
    #
    # MainWindow is now a pure widget container. run.py handles:
    #   - QApplication creation
    #   - showing the window
    #   - starting the event loop
