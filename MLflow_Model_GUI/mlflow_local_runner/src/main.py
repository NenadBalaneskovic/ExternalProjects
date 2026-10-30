# mlflow_local_runner/src/main.py
"""
main.py – Einstiegspunkt der MLflow Local Runner GUI

Dieses Modul initialisiert:
- Logging
- Konfiguration
- Die PySide6/PyQt5 GUI
- Das Hauptfenster (AppWindow)

Es wird automatisch gestartet, wenn der Nutzer im Terminal eingibt:
    mlflow-local-runner
oder
    python -m mlflow_local_runner
"""

import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from gui.app_window import AppWindow
from utils.logger import get_logger
from core.config_loader import ConfigLoader


def main():
    """Startpunkt der gesamten Anwendung."""
    logger = get_logger(__name__)
    logger.info("Starte MLflow Local Runner GUI...")

    # QApplication initialisieren
    app = QApplication(sys.argv)

    # Konfiguration laden (Tracking-URI, Artefaktpfad, letzte Pfade etc.)
    try:
        config = ConfigLoader().load()
        logger.info("Konfiguration erfolgreich geladen.")
    except Exception as e:
        logger.error(f"Fehler beim Laden der Konfiguration: {e}")
        config = {}

    # Hauptfenster erzeugen
    try:
        window = AppWindow(config=config)
        window.show()
        logger.info("GUI erfolgreich initialisiert.")
    except Exception as e:
        logger.error("Fehler beim Initialisieren des Hauptfensters.")
        logger.error(traceback.format_exc())
        QMessageBox.critical(
            None,
            "Fehler beim Starten",
            f"Die GUI konnte nicht gestartet werden:\n\n{e}"
        )
        sys.exit(1)

    # Event Loop starten
    try:
        sys.exit(app.exec())
    except Exception as e:
        logger.error("Unerwarteter Fehler im GUI-Eventloop.")
        logger.error(traceback.format_exc())
        QMessageBox.critical(
            None,
            "Laufzeitfehler",
            f"Ein unerwarteter Fehler ist aufgetreten:\n\n{e}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()