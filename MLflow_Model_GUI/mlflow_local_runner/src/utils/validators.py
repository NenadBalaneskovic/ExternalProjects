# mlflow_local_runner/src/utils/validators.py
"""
validators.py – Validierungsfunktionen für MLflow Local Runner

Dieses Modul stellt Funktionen bereit für:
- Datei-Validierung (.py, .csv)
- Ordner-Validierung
- URI-Validierung
"""

import os
from pathlib import Path
from urllib.parse import urlparse


# ---------------------------------------------------------
# GENERISCHE VALIDIERUNGEN
# ---------------------------------------------------------

def validate_file(path: str) -> bool:
    """Prüft, ob eine Datei existiert und lesbar ist."""
    if not path:
        return False
    p = Path(path)
    return p.exists() and p.is_file()


def validate_directory(path: str) -> bool:
    """Prüft, ob ein Ordner existiert."""
    if not path:
        return False
    p = Path(path)
    return p.exists() and p.is_dir()


# ---------------------------------------------------------
# DATEITYPEN
# ---------------------------------------------------------

def validate_python_file(path: str) -> bool:
    """Prüft, ob eine gültige .py-Datei existiert."""
    return validate_file(path) and path.lower().endswith(".py")


def validate_csv_file(path: str) -> bool:
    """Prüft, ob eine gültige .csv-Datei existiert."""
    return validate_file(path) and path.lower().endswith(".csv")


# ---------------------------------------------------------
# URI VALIDIERUNG
# ---------------------------------------------------------

def validate_uri(uri: str) -> bool:
    """
    Prüft, ob eine URI syntaktisch korrekt ist.
    Beispiele:
        http://localhost:5000
        https://example.com
    """
    if not uri:
        return False

    parsed = urlparse(uri)
    return all([parsed.scheme, parsed.netloc])
