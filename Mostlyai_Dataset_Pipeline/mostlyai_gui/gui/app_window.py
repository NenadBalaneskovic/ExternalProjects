"""
app_window.py

Defines the main application window for the Data Privacy Workbench GUI.
This window hosts all functional tabs:
    - Data Cleaning
    - Anonymization
    - Pseudonymization
    - Synthetic Data Generation
    - Logs

It also manages:
    - Global logging
    - Status bar updates
    - Cross-tab communication
    - Configuration loading (paths, API key, preferences)

Author: Nenad
Date: May 2026
"""

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QStatusBar,
)
from PySide6.QtGui import QIcon
from PySide6.QtCore import Signal, Slot

# Import tab widgets
from gui.tabs.cleaning_tab import CleaningTab
from gui.tabs.anonymization_tab import AnonymizationTab
from gui.tabs.pseudoanonymization_tab import PseudonymizationTab
from gui.tabs.synthetic_tab import SyntheticTab
from gui.tabs.logs_tab import LogsTab


class AppWindow(QMainWindow):
    """
    Main application window that orchestrates all GUI components.

    Parameters
    ----------
    config : dict
        Loaded configuration (paths, API key, preferences).
    logger : logging.Logger
        Central logger instance shared across the entire application.
    """

    # Signals for cross-tab communication
    cleaned_dataset_ready = Signal(object)          # pandas.DataFrame
    anonymized_dataset_ready = Signal(object)       # pandas.DataFrame
    pseudonymized_dataset_ready = Signal(object)    # pandas.DataFrame
    synthetic_dataset_ready = Signal(object)        # pandas.DataFrame

    log_message = Signal(str)                       # Forward logs to GUI

    def __init__(self, config: dict, logger):
        super().__init__()

        self.config = config
        self.logger = logger

        self.setWindowTitle("Data Privacy Workbench")
        self.setWindowIcon(QIcon("assets/icons/settings.png"))
        self.resize(1200, 800)

        # --- Central layout ---
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)

        # --- Tab widget ---
        self.tabs = QTabWidget()
        central_layout.addWidget(self.tabs)

        # --- Status bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # --- Initialize tabs ---
        self._init_tabs()

        # --- Connect logging ---
        self.log_message.connect(self._append_log_to_logs_tab)

        # --- Finalize ---
        self.setCentralWidget(central_widget)
        self.logger.info("AppWindow initialized.")

    # ------------------------------------------------------------------
    # Tab initialization
    # ------------------------------------------------------------------

    def _init_tabs(self):
        """Create and add all tabs to the tab widget."""

        # Data Cleaning
        self.cleaning_tab = CleaningTab(
            config=self.config,
            logger=self.logger,
            status_callback=self.update_status,
            log_callback=self.forward_log,
        )
        self.cleaning_tab.cleaned_dataset_ready.connect(self.cleaned_dataset_ready)
        self.tabs.addTab(self.cleaning_tab, QIcon(), "Data Cleaning")

        # Anonymization
        self.anonymization_tab = AnonymizationTab(
            config=self.config,
            logger=self.logger,
            status_callback=self.update_status,
            log_callback=self.forward_log,
        )
        self.cleaned_dataset_ready.connect(self.anonymization_tab.receive_cleaned_dataset)
        self.tabs.addTab(self.anonymization_tab, QIcon(), "Anonymization")

        # Pseudonymization
        self.pseudonymization_tab = PseudonymizationTab(
            config=self.config,
            logger=self.logger,
            status_callback=self.update_status,
            log_callback=self.forward_log,
        )
        self.cleaned_dataset_ready.connect(self.pseudonymization_tab.receive_cleaned_dataset)
        self.tabs.addTab(self.pseudonymization_tab, QIcon(), "Pseudonymization")

        # Synthetic Data Generation
        self.synthetic_tab = SyntheticTab(
            config=self.config,
            logger=self.logger,
            status_callback=self.update_status,
            log_callback=self.forward_log,
        )
        self.cleaned_dataset_ready.connect(self.synthetic_tab.receive_cleaned_dataset)
        self.tabs.addTab(self.synthetic_tab, QIcon(), "Synthetic Data")

        # Logs
        self.logs_tab = LogsTab()
        self.tabs.addTab(self.logs_tab, QIcon(), "Logs")

    # ------------------------------------------------------------------
    # Logging and status updates
    # ------------------------------------------------------------------

    @Slot(str)
    def forward_log(self, message: str):
        """
        Forward a log message from any tab to the Logs tab and status bar.

        Parameters
        ----------
        message : str
            Log message to display.
        """
        self.log_message.emit(message)
        self.update_status(message)

    @Slot(str)
    def _append_log_to_logs_tab(self, message: str):
        """Append a log message to the Logs tab."""
        self.logs_tab.append_log(message)

    @Slot(str)
    def update_status(self, message: str):
        """Update the status bar with a short message."""
        self.status_bar.showMessage(message, 5000)  # auto-clear after 5 seconds
