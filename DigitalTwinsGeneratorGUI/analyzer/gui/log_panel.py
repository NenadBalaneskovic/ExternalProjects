# analyzer/gui/log_panel.py

from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QTextEdit
from PySide6.QtCore import Qt
from datetime import datetime


class LogPanel(QWidget):
    """
    Log panel for the Telemetry Analyzer.

    Responsibilities:
        - Display system logs, alerts, and module messages
        - Provide an append-only text console
        - Timestamp each entry for clarity

    Methods:
        append_log(text: str)
            Appends a timestamped log entry to the console
    """

    def __init__(self):
        super().__init__()
        self._build_ui()

    # ---------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()

        group = QGroupBox("System Log")
        group_layout = QVBoxLayout()

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setLineWrapMode(QTextEdit.NoWrap)

        group_layout.addWidget(self.text_area)
        group.setLayout(group_layout)

        layout.addWidget(group)
        self.setLayout(layout)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def append_log(self, text: str):
        """
        Appends a timestamped log entry to the console.

        Args:
            text (str): Log message
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {text}"

        self.text_area.append(entry)
        self.text_area.ensureCursorVisible()