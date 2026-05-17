# mlflow_local_runner/src/utils/paths.py
"""
paths.py – Zentrale Pfadverwaltung für MLflow Local Runner

Dieses Modul stellt Funktionen bereit für:
- Konfigurationsordner
- Artefaktordner
- Logordner

Die Pfade sind plattformübergreifend (Windows, Linux, macOS)
und werden automatisch erstellt.
"""

import os
from pathlib import Path


# ---------------------------------------------------------
# BASIS-ANWENDUNGSORDNER
# ---------------------------------------------------------

def get_base_app_dir() -> Path:
    """
    Liefert den Basisordner für alle persistenten Daten.
    Windows:   C:/Users/<User>/AppData/Roaming/mlflow_local_runner/
    Linux:     ~/.config/mlflow_local_runner/
    macOS:     ~/Library/Application Support/mlflow_local_runner/
    """
    if os.name == "nt":  # Windows
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "mlflow_local_runner"

    elif sys.platform == "darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "mlflow_local_runner"

    else:  # Linux / Unix
        return Path.home() / ".config" / "mlflow_local_runner"


# ---------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------

def get_config_dir() -> Path:
    """
    Ordner für Konfigurationsdateien.
    """
    path = get_base_app_dir() / "config"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------
# ARTEFAKTE
# ---------------------------------------------------------

def get_artifact_base_dir() -> Path:
    """
    Basisordner für Artefakte.
    """
    path = get_base_app_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

def get_log_dir() -> Path:
    """
    Ordner für Log-Dateien.
    """
    path = get_base_app_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
