# analyzer/gui/health_summary.py

from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QLabel, QHBoxLayout
from PySide6.QtCore import Qt
from datetime import datetime


class HealthSummary(QWidget):
    """
    Compact health indicator panel for the Telemetry Analyzer.

    Responsibilities:
        - Display system health status (OK / Warning / Error)
        - Show last update time
        - React to alerts from the Generator
        - React to analysis results (e.g., anomalies)

    Methods:
        update_health(summary: dict)
            Called by AnalyzerLoop with module-derived health info

        mark_data_updated()
            Called when new data arrives (e.g., chunk_written alert)

        mark_generation_complete()
            Called when generator finishes producing data
    """

    def __init__(self):
        super().__init__()
        self._build_ui()

        # Internal state
        self.last_update = None
        self.status = "OK"

    # ---------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()

        group = QGroupBox("System Health")
        group_layout = QVBoxLayout()

        # Status label (color-coded)
        self.status_label = QLabel("Status: OK")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._set_status_color("OK")

        # Last update timestamp
        self.update_label = QLabel("Last update: —")
        self.update_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Generator state
        self.generator_label = QLabel("Generator: Active")
        self.generator_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        group_layout.addWidget(self.status_label)
        group_layout.addWidget(self.update_label)
        group_layout.addWidget(self.generator_label)

        group.setLayout(group_layout)
        layout.addWidget(group)
        self.setLayout(layout)

    # ---------------------------------------------------------
    # Status Color Helper
    # ---------------------------------------------------------
    def _set_status_color(self, status: str):
        """
        Applies color coding based on status.
        """
        if status == "OK":
            color = "green"
        elif status == "Warning":
            color = "orange"
        else:
            color = "red"

        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def update_health(self, summary: dict):
        """
        Called by AnalyzerLoop with health information.

        Expected summary format:
            {
                "status": "OK" | "Warning" | "Error",
                "message": "Optional description"
            }
        """
        status = summary.get("status", "OK")
        message = summary.get("message", "")

        self.status = status
        self._set_status_color(status)

        if message:
            self.status_label.setText(f"Status: {status} — {message}")
        else:
            self.status_label.setText(f"Status: {status}")

        # Update timestamp
        self.last_update = datetime.utcnow()
        self.update_label.setText(f"Last update: {self.last_update.strftime('%H:%M:%S')}")

    def mark_data_updated(self):
        """
        Called when new data arrives (e.g., chunk_written alert).
        """
        self.last_update = datetime.utcnow()
        self.update_label.setText(f"Last update: {self.last_update.strftime('%H:%M:%S')}")

    def mark_generation_complete(self):
        """
        Called when generator finishes producing data.
        """
        self.generator_label.setText("Generator: Complete")
        self.generator_label.setStyleSheet("color: blue; font-weight: bold;")