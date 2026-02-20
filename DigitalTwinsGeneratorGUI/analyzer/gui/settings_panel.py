# analyzer/gui/settings_panel.py

from PySide6.QtWidgets import (
    QWidget,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QLineEdit,
    QPushButton,
    QFileDialog,
)
from PySide6.QtCore import Qt, Signal


class SettingsPanel(QWidget):
    """
    Settings panel for the Telemetry Analyzer.

    Responsibilities:
        - Display file path (from config.json or user override)
        - Allow user to change refresh interval (polling frequency)
        - Allow user to limit number of rows loaded per refresh
        - Provide Start / Stop buttons for analysis

    Signals:
        start_requested -> emitted when user presses Start
        stop_requested  -> emitted when user presses Stop

    Methods:
        get_analysis_config() -> dict
            Returns current analysis settings for AnalyzerLoop
    """

    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, config: dict):
        super().__init__()

        self.config = config or {}

        self._build_ui()

    # ---------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()

        # --- File Path ---
        file_group = QGroupBox("Telemetry File")
        file_layout = QVBoxLayout()

        self.file_path_edit = QLineEdit()

        # Load from config["output"]["file_path"]
        default_path = (
            self.config.get("output", {}).get("file_path", "")
        )
        self.file_path_edit.setText(default_path)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_file)

        file_layout.addWidget(QLabel("File path:"))
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(browse_btn)
        file_group.setLayout(file_layout)

        # --- Refresh Interval ---
        refresh_group = QGroupBox("Refresh Settings")
        refresh_layout = QVBoxLayout()

        self.refresh_spin = QSpinBox()
        self.refresh_spin.setRange(100, 10_000)
        self.refresh_spin.setValue(500)  # ms
        self.refresh_spin.setSuffix(" ms")

        refresh_layout.addWidget(QLabel("Refresh interval:"))
        refresh_layout.addWidget(self.refresh_spin)
        refresh_group.setLayout(refresh_layout)

        # --- Row Limit ---
        row_group = QGroupBox("Row Processing")
        row_layout = QVBoxLayout()

        self.row_limit_spin = QSpinBox()
        self.row_limit_spin.setRange(100, 1_000_000)
        self.row_limit_spin.setValue(10_000)

        row_layout.addWidget(QLabel("Rows per refresh:"))
        row_layout.addWidget(self.row_limit_spin)
        row_group.setLayout(row_layout)

        # --- Start / Stop Buttons ---
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Analysis")
        self.stop_btn = QPushButton("Stop")

        self.start_btn.clicked.connect(self.start_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        # Assemble layout
        layout.addWidget(file_group)
        layout.addWidget(refresh_group)
        layout.addWidget(row_group)
        layout.addLayout(btn_layout)
        layout.addStretch()

        self.setLayout(layout)

    # ---------------------------------------------------------
    # File Browser
    # ---------------------------------------------------------
    def _browse_file(self):
        """
        Opens a file dialog to select a CSV or Parquet file.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Telemetry File",
            "",
            "Data Files (*.csv *.parquet)"
        )
        if path:
            self.file_path_edit.setText(path)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def get_analysis_config(self) -> dict:
        """
        Returns a dictionary with current analysis settings.
        Used by AnalyzerLoop.start().
        """
        return {
            "file_path": self.file_path_edit.text(),
            "refresh_interval_ms": self.refresh_spin.value(),
            "row_limit": self.row_limit_spin.value(),
        }