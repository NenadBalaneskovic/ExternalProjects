# mlflow_local_runner/src/gui/run_panel.py
"""
run_panel.py – Panel zum Starten eines MLflow-Runs

Dieses Panel enthält:
- Einen "Run starten"-Button
- Ein Textfeld für Live-Logs (mit Auto-Scroll)
- Signal/Slot-Mechanismen zur Kommunikation mit AppWindow

AppWindow übernimmt:
- Validierung (Dataset vorhanden)
- Übergabe der MLflow-Konfiguration
- Aufruf von Runner.run()
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
)
from PySide6.QtCore import Signal, Slot, Qt

from utils.logger import get_logger


class RunPanel(QWidget):
    """
    Panel zum Starten eines Runs und Anzeigen von Logs.
    """

    # Signal an AppWindow: Nutzer möchte einen Run starten
    start_run_clicked = Signal()

    def __init__(self):
        super().__init__()

        self.logger = get_logger(__name__)
        self._init_ui()

    # ---------------------------------------------------------
    # UI INITIALISIERUNG
    # ---------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Run starten")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Start-Button
        self.btn_start = QPushButton("Run starten")
        self.btn_start.setStyleSheet("font-size: 16px; padding: 8px;")
        layout.addWidget(self.btn_start)

        # Log-Ausgabe
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("""
            background-color: #1e1e1e;
            color: #dcdcdc;
            font-family: Consolas, monospace;
            font-size: 13px;
        """)
        layout.addWidget(self.log_output)

        self.setLayout(layout)

        # Button-Signal verbinden
        self.btn_start.clicked.connect(self._on_start_clicked)

    # ---------------------------------------------------------
    # RUN STARTEN
    # ---------------------------------------------------------

    @Slot()
    def _on_start_clicked(self):
        """Wird aufgerufen, wenn der Nutzer auf 'Run starten' klickt."""
        self.clear_log()
        self.logger.info("Run-Start angefordert.")
        self.append_log("<span style='color:#6ab04c;'>Starte Run...</span>")
        self.start_run_clicked.emit()

    # ---------------------------------------------------------
    # LOGGING
    # ---------------------------------------------------------

    def append_log(self, text: str):
        """
        Fügt Text zur Log-Ausgabe hinzu (mit Auto-Scroll).
        HTML wird unterstützt.
        """
        self.log_output.append(text)
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def append_info(self, text: str):
        self.append_log(f"<span style='color:#dcdcdc;'>{text}</span>")

    def append_warning(self, text: str):
        self.append_log(f"<span style='color:#f6e58d;'>[WARN] {text}</span>")

    def append_error(self, text: str):
        self.append_log(f"<span style='color:#ff7979;'>[ERROR] {text}</span>")

    def clear_log(self):
        """Leert das Log-Feld."""
        self.log_output.clear()

    # ---------------------------------------------------------
    # BUTTON-STEUERUNG
    # ---------------------------------------------------------

    def set_running(self, running: bool):
        """
        Aktiviert/Deaktiviert den Start-Button.
        Wird von AppWindow oder Runner aufgerufen.
        """
        self.btn_start.setEnabled(not running)
        if running:
            self.append_info("Run läuft...")
        else:
            self.append_info("Run beendet.")
