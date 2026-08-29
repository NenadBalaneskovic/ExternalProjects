"""
gui/log_panel.py

Log viewer panel for PyDependencyInspector.

Responsibilities:
- Display application logs in real time.
- Attach to the InMemoryLogBuffer via a callback.
- Provide simple controls (clear, auto-scroll).
- Apply dark graphite styling.

This module is intentionally:
- GUI-only.
- Connected to the global logging configuration.
"""

from __future__ import annotations

import re

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QCheckBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

from ..utils.logging_utils import configure_logging, get_logger


ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")


class LogPanel(QWidget):
    """
    Displays logs in a QTextEdit.
    """

    def __init__(self, project_state) -> None:
        super().__init__()
        self.project_state = project_state

        self._init_ui()
        self._apply_styles()
        self._attach_logger()

    # ------------------------------------------------------------------ #
    # UI initialization
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear)

        self.chk_autoscroll = QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)

        controls.addWidget(self.btn_clear)
        controls.addWidget(self.chk_autoscroll)
        controls.addStretch(1)

        self.text = QTextEdit()
        self.text.setReadOnly(True)

        layout.addLayout(controls)
        layout.addWidget(self.text)

    # ------------------------------------------------------------------ #
    # Styling
    # ------------------------------------------------------------------ #

    def _apply_styles(self) -> None:
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #333;
                font-family: Consolas, "Fira Code", monospace;
                font-size: 12px;
            }
            QPushButton {
                background-color: #2b2b2b;
                color: #e0e0e0;
                padding: 4px 10px;
                border-radius: 4px;
                border: 1px solid #444;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QCheckBox {
                color: #e0e0e0;
            }
        """)

    # ------------------------------------------------------------------ #
    # Logging integration
    # ------------------------------------------------------------------ #

    def _attach_logger(self) -> None:
        """
        Configure global logging and attach our callback to the in-memory buffer.
        """
        def callback(msg: str) -> None:
            self.append_line(msg)

        self._buffer = configure_logging(gui_callback=callback)
        self._logger = get_logger(__name__)

    def append_line(self, line: str) -> None:
        """
        Append a single log line to the view.
        Strip ANSI escape codes for clean display.
        """
        clean_line = ANSI_ESCAPE_RE.sub("", line)
        self.text.append(clean_line)

        if self.chk_autoscroll.isChecked():
            self.text.moveCursor(QTextCursor.End)

    def clear(self) -> None:
        """
        Clear the log view and the in-memory buffer.
        """
        self.text.clear()
        if hasattr(self, "_buffer"):
            self._buffer.clear()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_clean_logs(self) -> str:
        """
        Return logs as clean plain text (no ANSI codes).
        """
        raw = self.text.toPlainText()
        return ANSI_ESCAPE_RE.sub("", raw)
