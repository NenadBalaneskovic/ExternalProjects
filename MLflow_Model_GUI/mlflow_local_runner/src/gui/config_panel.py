# mlflow_local_runner/src/gui/config_panel.py
"""
config_panel.py – MLflow-Konfigurationspanel (tokenfrei)

Dieses Panel erlaubt dem Nutzer:
- Tracking-URI festzulegen
- Registry-URI festzulegen
- Artefakt-Ordner auszuwählen
- Projektname festzulegen
- Experimentname festzulegen
- Runname festzulegen
- Konfiguration zu speichern
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QFileDialog, QMessageBox
)
from PySide6.QtCore import Signal

from core.config_loader import ConfigLoader
from utils.validators import validate_uri, validate_directory
from utils.logger import get_logger


class ConfigPanel(QWidget):

    config_saved = Signal(dict)

    def __init__(self, initial_config: dict | None = None):
        super().__init__()

        self.logger = get_logger(__name__)
        self.config_loader = ConfigLoader()

        self.config = initial_config or {}

        self._init_ui()
        self._load_initial_values()

    # ---------------------------------------------------------
    # UI INITIALISIERUNG
    # ---------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout()
        form = QFormLayout()

        # Bestehende Felder
        self.tracking_uri_input = QLineEdit()
        self.registry_uri_input = QLineEdit()
        self.artifact_dir_input = QLineEdit()

        # Neue Felder
        self.project_name_input = QLineEdit()
        self.experiment_name_input = QLineEdit()
        self.run_name_input = QLineEdit()

        # Buttons
        self.btn_select_artifact_dir = QPushButton("Ordner auswählen")
        self.btn_save = QPushButton("Konfiguration speichern")

        # Form zusammenbauen
        form.addRow("Projektname:", self.project_name_input)
        form.addRow("Experimentname:", self.experiment_name_input)
        form.addRow("Runname:", self.run_name_input)

        form.addRow("MLflow Tracking URI:", self.tracking_uri_input)
        form.addRow("MLflow Registry URI:", self.registry_uri_input)
        form.addRow("Artefakt-Ordner:", self.artifact_dir_input)
        form.addRow("", self.btn_select_artifact_dir)

        layout.addLayout(form)
        layout.addWidget(self.btn_save)
        self.setLayout(layout)

        # Signale verbinden
        self.btn_select_artifact_dir.clicked.connect(self._select_artifact_dir)
        self.btn_save.clicked.connect(self._save_config)

    # ---------------------------------------------------------
    # INITIALWERTE LADEN
    # ---------------------------------------------------------

    def _load_initial_values(self):
        self.project_name_input.setText(self.config.get("project_name", ""))
        self.experiment_name_input.setText(self.config.get("experiment_name", ""))
        self.run_name_input.setText(self.config.get("run_name", ""))

        self.tracking_uri_input.setText(self.config.get("tracking_uri", "http://localhost:5000"))
        self.registry_uri_input.setText(self.config.get("registry_uri", "http://localhost:5000"))
        self.artifact_dir_input.setText(self.config.get("artifact_dir", ""))

    # ---------------------------------------------------------
    # ARTIFAKT-ORDNER AUSWÄHLEN
    # ---------------------------------------------------------

    def _select_artifact_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Artefakt-Ordner auswählen")
        if directory:
            self.artifact_dir_input.setText(directory)

    # ---------------------------------------------------------
    # KONFIGURATION SPEICHERN
    # ---------------------------------------------------------

    def _save_config(self):
        tracking_uri = self.tracking_uri_input.text().strip()
        registry_uri = self.registry_uri_input.text().strip()
        artifact_dir = self.artifact_dir_input.text().strip()

        project_name = self.project_name_input.text().strip()
        experiment_name = self.experiment_name_input.text().strip()
        run_name = self.run_name_input.text().strip()

        # Validierung
        if not validate_uri(tracking_uri):
            QMessageBox.warning(self, "Ungültige URI", "Die Tracking-URI ist ungültig.")
            return

        if registry_uri and not validate_uri(registry_uri):
            QMessageBox.warning(self, "Ungültige URI", "Die Registry-URI ist ungültig.")
            return

        if artifact_dir and not validate_directory(artifact_dir):
            QMessageBox.warning(self, "Ungültiger Ordner", "Der Artefakt-Ordner ist ungültig.")
            return

        # Konfiguration speichern
        new_config = {
            "tracking_uri": tracking_uri,
            "registry_uri": registry_uri,
            "artifact_dir": artifact_dir,
            "project_name": project_name,
            "experiment_name": experiment_name,
            "run_name": run_name,
        }

        try:
            self.config_loader.save(new_config)
            self.logger.info("Konfiguration gespeichert.")
            self.config_saved.emit(new_config)
            QMessageBox.information(self, "Gespeichert", "Konfiguration erfolgreich gespeichert.")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Konfiguration: {e}")
            QMessageBox.critical(self, "Fehler", f"Konfiguration konnte nicht gespeichert werden:\n{e}")

    # ---------------------------------------------------------
    # KONFIGURATION ABRUFEN
    # ---------------------------------------------------------

    def get_config(self) -> dict:
        return {
            "tracking_uri": self.tracking_uri_input.text().strip(),
            "registry_uri": self.registry_uri_input.text().strip(),
            "artifact_dir": self.artifact_dir_input.text().strip(),
            "project_name": self.project_name_input.text().strip(),
            "experiment_name": self.experiment_name_input.text().strip(),
            "run_name": self.run_name_input.text().strip(),
        }
