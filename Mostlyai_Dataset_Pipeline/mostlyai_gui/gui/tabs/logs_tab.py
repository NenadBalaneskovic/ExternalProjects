"""
logs_tab.py

Implements the Logs tab of the Data Privacy Workbench GUI.
This tab provides:

    - A scrollable log viewer
    - Filtering by log level (INFO, WARNING, ERROR)
    - Filtering by module (Cleaning, Anonymization, Pseudonymization, Synthetic, System)
    - Exporting logs to a text file

The tab receives log messages from AppWindow via append_log().

Author: Nenad
Date: May 2026
"""

import os
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QFileDialog,
    QTextEdit,
)
from PySide6.QtGui import QTextCursor
from PySide6.QtCore import Qt


class LogsTab(QWidget):
    """
    Logs tab widget.

    Provides a full log viewer with filtering and export functionality.
    """

    def __init__(self):
        super().__init__()

        self.all_logs = []  # store raw log messages
        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --------------------------------------------------------------
        # Filter controls
        # --------------------------------------------------------------
        filter_layout = QHBoxLayout()

        # Log level filter
        filter_layout.addWidget(QLabel("Level:"))
        self.level_filter = QComboBox()
        self.level_filter.addItems(["All", "INFO", "WARNING", "ERROR"])
        self.level_filter.currentIndexChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.level_filter)

        # Module filter
        filter_layout.addWidget(QLabel("Module:"))
        self.module_filter = QComboBox()
        self.module_filter.addItems([
            "All",
            "Cleaning",
            "Anonymization",
            "Pseudonymization",
            "Synthetic",
            "System",
        ])
        self.module_filter.currentIndexChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.module_filter)

        # Export button
        export_btn = QPushButton("Export Log…")
        export_btn.clicked.connect(self._export_log)
        filter_layout.addWidget(export_btn)

        filter_layout.addStretch()

        # --------------------------------------------------------------
        # Log viewer
        # --------------------------------------------------------------
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.NoWrap)

        layout.addLayout(filter_layout)
        layout.addWidget(self.log_view)

    # ------------------------------------------------------------------
    # Log handling
    # ------------------------------------------------------------------

    def append_log(self, message: str):
        """Append a new log message to the viewer and internal buffer."""
        self.all_logs.append(message)
        self._apply_filters()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _apply_filters(self):
        """Apply level and module filters to the log viewer."""
        level_filter = self.level_filter.currentText()
        module_filter = self.module_filter.currentText()

        filtered = []

        for msg in self.all_logs:
            msg_upper = msg.upper()

            # --- Level filter (robust) ---
            if level_filter != "All":
                if level_filter == "WARNING":
                    # Accept both WARNING and WARN
                    if "WARNING" not in msg_upper and "WARN" not in msg_upper:
                        continue
                else:
                    if level_filter not in msg_upper:
                        continue

            # --- Module filter ---
            if module_filter != "All" and module_filter.upper() not in msg_upper:
                continue

            filtered.append(msg)

        # Update viewer
        self.log_view.clear()
        self.log_view.append("\n".join(filtered))

        # Auto-scroll to bottom
        self.log_view.moveCursor(QTextCursor.End)

    # ------------------------------------------------------------------
    # Export logs
    # ------------------------------------------------------------------

    def _export_log(self):
        """Export the full log buffer to a text file."""
        default_name = f"log_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Log",
            default_name,
            "Text Files (*.txt)"
        )

        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.all_logs))

        self.log_view.append(f"\n[INFO] System: Log exported to {path}")
        self.log_view.moveCursor(QTextCursor.End)
