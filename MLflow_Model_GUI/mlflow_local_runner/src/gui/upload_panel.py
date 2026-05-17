# mlflow_local_runner/src/gui/upload_panel.py
"""
upload_panel.py – Datei-Upload-Panel für Skript & Dataset

Dieses Panel erlaubt dem Nutzer:
- Ein Trainingsskript (.py) auszuwählen (optional)
- Ein Dataset (.csv) auszuwählen (Pflicht)
- Validierung der Dateien
- Übergabe der Pfade an AppWindow über Qt-Signale
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QHBoxLayout, QLineEdit, QMessageBox
)
from PySide6.QtCore import Signal, Slot, Qt

from utils.validators import (
    validate_python_file, validate_csv_file
)
from utils.logger import get_logger


class UploadPanel(QWidget):
    """
    Panel für Datei-Uploads:
    - Nutzer-Skript (.py)
    - Dataset (.csv)
    """

    script_selected = Signal(str)
    dataset_selected = Signal(str)

    def __init__(self):
        super().__init__()

        self.logger = get_logger(__name__)
        self.script_path = None
        self.dataset_path = None

        self._init_ui()

    # ---------------------------------------------------------
    # UI INITIALISIERUNG
    # ---------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Dateien hochladen")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # -----------------------------------------------------
        # Skript-Auswahl
        # -----------------------------------------------------
        script_layout = QHBoxLayout()
        self.script_input = QLineEdit()
        self.script_input.setPlaceholderText("Kein Skript ausgewählt (Template wird verwendet)")
        self.script_input.setReadOnly(True)

        btn_script = QPushButton("Skript auswählen (.py)")
        btn_script.clicked.connect(self._select_script)

        script_layout.addWidget(self.script_input)
        script_layout.addWidget(btn_script)
        layout.addLayout(script_layout)

        # -----------------------------------------------------
        # Dataset-Auswahl
        # -----------------------------------------------------
        dataset_layout = QHBoxLayout()
        self.dataset_input = QLineEdit()
        self.dataset_input.setPlaceholderText("Bitte Dataset auswählen (.csv)")
        self.dataset_input.setReadOnly(True)

        btn_dataset = QPushButton("Dataset auswählen (.csv)")
        btn_dataset.clicked.connect(self._select_dataset)

        dataset_layout.addWidget(self.dataset_input)
        dataset_layout.addWidget(btn_dataset)
        layout.addLayout(dataset_layout)

        self.setLayout(layout)

    # ---------------------------------------------------------
    # SKRIPT AUSWÄHLEN
    # ---------------------------------------------------------

    @Slot()
    def _select_script(self):
        """Öffnet Dateidialog für .py-Skripte."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Trainingsskript auswählen",
            "",
            "Python-Skripte (*.py)"
        )

        if not path:
            return

        if not validate_python_file(path):
            QMessageBox.warning(self, "Ungültige Datei",
                                "Bitte wählen Sie eine gültige .py-Datei aus.")
            return

        self.script_path = path
        self.script_input.setText(path)
        self.logger.info(f"Skript ausgewählt: {path}")

        # Signal an AppWindow
        self.script_selected.emit(path)

    # ---------------------------------------------------------
    # DATASET AUSWÄHLEN
    # ---------------------------------------------------------

    @Slot()
    def _select_dataset(self):
        """Öffnet Dateidialog für .csv-Datasets."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Dataset auswählen",
            "",
            "CSV-Dateien (*.csv)"
        )

        if not path:
            return

        if not validate_csv_file(path):
            QMessageBox.warning(self, "Ungültige Datei",
                                "Bitte wählen Sie eine gültige .csv-Datei aus.")
            return

        self.dataset_path = path
        self.dataset_input.setText(path)
        self.logger.info(f"Dataset ausgewählt: {path}")

        # Signal an AppWindow
        self.dataset_selected.emit(path)
