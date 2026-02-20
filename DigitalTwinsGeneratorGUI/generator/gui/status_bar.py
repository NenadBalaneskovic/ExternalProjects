# generator/gui/status_bar.py

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt


class StatusBar(QWidget):
    """
    Status bar for the Telemetry Generator GUI.
    Displays:
        - Progress bar (file size or rows written)
        - Status message (e.g., 'Generating...', 'Chunk written')
        - Alerts from the generator backend

    Methods:
        update_progress(percent, message)
        show_message(text)
        show_alert(text)
    """

    def __init__(self):
        super().__init__()

        self._build_ui()

    # ---------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------
    def _build_ui(self):
        layout = QHBoxLayout()

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("0%")
        self.progress.setTextVisible(True)

        # Status message label
        self.message_label = QLabel("Ready.")
        self.message_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Alert label
        self.alert_label = QLabel("")
        self.alert_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.alert_label.setStyleSheet("color: red; font-weight: bold;")

        layout.addWidget(self.progress, stretch=2)
        layout.addWidget(self.message_label, stretch=3)
        layout.addWidget(self.alert_label, stretch=2)

        self.setLayout(layout)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def update_progress(self, percent: float, message: str = ""):
        """
        Updates the progress bar and optional message.

        Args:
            percent (float): 0–100 progress value
            message (str): Optional status message
        """
        p = int(percent)
        self.progress.setValue(p)

        # Show percentage inside the bar
        self.progress.setFormat(f"{p}%")

        # Optional external message label
        if message:
            self.message_label.setText(message)

    def show_message(self, text: str):
        """
        Shows a neutral status message.
        """
        self.message_label.setText(text)
        self.alert_label.setText("")

    def show_alert(self, text: str):
        """
        Shows an alert message (e.g., from alert socket).
        """
        self.alert_label.setText(text)