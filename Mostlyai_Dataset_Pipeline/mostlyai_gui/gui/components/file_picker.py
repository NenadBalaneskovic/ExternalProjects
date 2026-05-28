"""
file_picker.py

A reusable file picker widget for the Data Privacy Workbench GUI.

This component provides:
    - A label showing the currently selected file
    - A "Browse..." button to open a file dialog
    - Optional file type filters (default: CSV)
    - A callback that fires when a file is selected

It is intentionally lightweight and reusable across multiple tabs:
    - CleaningTab
    - AnonymizationTab
    - PseudonymizationTab
    - SyntheticTab

Author: Nenad
Date: May 2026
"""

import os
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
)
from PySide6.QtCore import Signal


class FilePicker(QWidget):
    """
    A simple file picker widget with a label and a browse button.

    Signals
    -------
    file_selected : str
        Emitted when a file is selected. Contains the file path.
    """

    file_selected = Signal(str)

    def __init__(self, label_text="No file selected.", file_filter="CSV Files (*.csv)"):
        super().__init__()

        self.file_filter = file_filter
        self.file_path = None

        self._build_ui(label_text)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self, label_text):
        layout = QHBoxLayout(self)

        # Label showing selected file
        self.label = QLabel(label_text)
        self.label.setMinimumWidth(250)

        # Browse button
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._open_dialog)

        layout.addWidget(self.label)
        layout.addWidget(browse_btn)

    # ------------------------------------------------------------------
    # File dialog
    # ------------------------------------------------------------------

    def _open_dialog(self):
        """Open a file dialog and update the label when a file is selected."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            self.file_filter,
        )

        if not path:
            return

        self.file_path = path
        self.label.setText(os.path.basename(path))
        self.file_selected.emit(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_path(self) -> str | None:
        """Return the currently selected file path."""
        return self.file_path

    def set_default(self, path: str):
        """
        Set a default file path (e.g., sample dataset).
        Updates the label accordingly.
        """
        self.file_path = path
        self.label.setText(os.path.basename(path))
