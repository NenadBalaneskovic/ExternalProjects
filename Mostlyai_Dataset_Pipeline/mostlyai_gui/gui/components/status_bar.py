"""
status_bar.py

A reusable enhanced status bar widget for the Data Privacy Workbench GUI.

This component wraps QStatusBar and provides:
    - A clean API for showing status messages
    - Optional progress indicator (spinner or text)
    - Automatic timeout clearing
    - Consistent styling across the application

It is used by AppWindow to display:
    - Cleaning progress
    - Anonymization progress
    - Pseudonymization progress
    - Synthetic data generation progress
    - System messages

Author: Nenad
Date: May 2026
"""

from PySide6.QtWidgets import QStatusBar, QLabel, QProgressBar
from PySide6.QtCore import QTimer


class EnhancedStatusBar(QStatusBar):
    """
    A custom status bar with message display and optional progress indicator.

    Methods
    -------
    show_message(text, timeout=5000)
        Display a message for a given duration.
    start_progress()
        Show an indeterminate progress bar.
    stop_progress()
        Hide the progress bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Message label
        self.message_label = QLabel("")
        self.addWidget(self.message_label)

        # Progress bar (hidden by default)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate
        self.progress.setVisible(False)
        self.progress.setFixedWidth(150)
        self.addPermanentWidget(self.progress)

        # Timer for auto-clearing messages
        self.clear_timer = QTimer()
        self.clear_timer.setSingleShot(True)
        self.clear_timer.timeout.connect(self._clear_message)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_message(self, text: str, timeout: int = 5000):
        """
        Display a status message.

        Parameters
        ----------
        text : str
            Message to display.
        timeout : int
            Duration in milliseconds before clearing the message.
        """
        self.message_label.setText(text)

        if timeout > 0:
            self.clear_timer.start(timeout)

    def start_progress(self):
        """Show the indeterminate progress bar."""
        self.progress.setVisible(True)

    def stop_progress(self):
        """Hide the progress bar."""
        self.progress.setVisible(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clear_message(self):
        """Clear the message label."""
        self.message_label.setText("")
