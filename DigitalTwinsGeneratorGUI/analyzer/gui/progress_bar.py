# analyzer/gui/progress_bar.py

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt


class ProgressBar(QWidget):
    """
    Progress bar for the Telemetry Analyzer.

    Responsibilities:
        - Display progress of file processing
        - Show status messages (e.g., "Reading...", "Analyzing...")
        - Provide a clean, compact UI element for the bottom of MainWindow

    Methods:
        update_progress(percent, message)
            Updates the progress bar and status label
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

        # Status message
        self.message_label = QLabel("Idle.")
        self.message_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout.addWidget(self.progress, stretch=2)
        layout.addWidget(self.message_label, stretch=3)

        self.setLayout(layout)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def update_progress(self, percent: float, message: str = ""):
        """
        Updates the progress bar and optional status message.

        Args:
            percent (float): 0–100 progress value
            message (str): Optional status message
        """
        p = int(percent)

        # Update the bar value
        self.progress.setValue(p)

        # Show the percentage inside the bar
        self.progress.setFormat(f"{p}%")

        # Optional external message label
        if message:
            self.message_label.setText(message)
