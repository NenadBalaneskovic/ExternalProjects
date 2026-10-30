# mlflow_local_runner/src/core/config_loader.py
"""
config_loader.py – Laden und Speichern der GUI-Konfiguration

Dieses Modul verwaltet eine persistente JSON-Konfigurationsdatei.
Gespeichert werden u.a.:
- MLflow Tracking URI
- MLflow Registry URI
- Artefakt-Ordner
- Zuletzt verwendete Pfade (optional erweiterbar)

Die Datei wird automatisch erstellt, falls sie nicht existiert.
"""

import json
from pathlib import Path

from utils.logger import get_logger
from utils.paths import get_config_dir


class ConfigLoader:
    """
    Lädt und speichert die GUI-Konfiguration in einer JSON-Datei.
    """

    CONFIG_FILENAME = "config.json"

    def __init__(self):
        self.logger = get_logger(__name__)
        self.config_dir = Path(get_config_dir())
        self.config_path = self.config_dir / self.CONFIG_FILENAME

        # Ordner erstellen, falls nicht vorhanden
        self.config_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # KONFIGURATION LADEN
    # ---------------------------------------------------------

    def load(self) -> dict:
        """
        Lädt die Konfiguration aus der JSON-Datei.
        Falls die Datei nicht existiert, wird eine leere Konfiguration zurückgegeben.
        """

        if not self.config_path.exists():
            self.logger.info("Keine Konfigurationsdatei gefunden – Standardwerte werden verwendet.")
            return {}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.logger.info(f"Konfiguration geladen: {config}")
                return config
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Konfiguration: {e}")
            return {}

    # ---------------------------------------------------------
    # KONFIGURATION SPEICHERN
    # ---------------------------------------------------------

    def save(self, config: dict):
        """
        Speichert die Konfiguration in der JSON-Datei.
        """

        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Konfiguration gespeichert: {config}")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Konfiguration: {e}")
            raise RuntimeError(f"Konfiguration konnte nicht gespeichert werden: {e}")
