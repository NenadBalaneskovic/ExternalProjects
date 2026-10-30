"""
log_viewer.py

A reusable log viewer widget for the Data Privacy Workbench GUI.

This component provides:
    - A scrollable, read-only text area
    - Optional auto-scrolling to the newest log entry
    - A clean API for appending and clearing logs

It is intentionally lightweight and can be embedded anywhere:
    - Inside tabs
    - Inside dialogs
    - Inside debugging panels

Author: Nenad
Date: May 2026
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PySide6.QtCore import Qt


class LogViewer(QWidget):
    """
    A simple scrollable log viewer widget.

    Methods
    -------
    append(text)
        Append a new line of text to the log viewer.
    clear()
        Clear all log contents.
    set_autoscroll(enabled)
        Enable or disable automatic scrolling to the bottom.
    """

    def __init__(self, autoscroll=True):
        super().__init__()

        self.autoscroll = autoscroll
        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setLineWrapMode(QTextEdit.NoWrap)

        layout.addWidget(self.text_area)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, text: str):
        """
        Append a new log entry.

        Parameters
        ----------
        text : str
            The log message to append.
        """
        if not text.endswith("\n"):
            text += "\n"

        self.text_area.moveCursor(Qt.TextCursorEnd)
        self.text_area.insertPlainText(text)

        if self.autoscroll:
            self.text_area.moveCursor(Qt.TextCursorEnd)

    def clear(self):
        """Clear all log contents."""
        self.text_area.clear()

    def set_autoscroll(self, enabled: bool):
        """
        Enable or disable automatic scrolling.

        Parameters
        ----------
        enabled : bool
            True to enable auto-scrolling, False to disable it.
        """
        self.autoscroll = enabled
