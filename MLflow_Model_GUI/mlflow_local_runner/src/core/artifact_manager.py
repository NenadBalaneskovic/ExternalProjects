# mlflow_local_runner/src/core/artifact_manager.py
"""
artifact_manager.py – Verwaltung lokaler Artefakte für MLflow Local Runner

Dieses Modul übernimmt:
- Erstellen eines lokalen Artefakt-Ordners
- Erstellen eines Run-spezifischen Unterordners
- Speichern von Modellrepräsentationen, Logs und Debug-Dateien
- Aufräumen alter Artefakte
"""

import shutil
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger
from utils.paths import get_artifact_base_dir


class ArtifactManager:
    """
    Verwaltet lokale Artefakte für jeden Run.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.base_dir = Path(get_artifact_base_dir())
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # RUN-ORDNER ERSTELLEN
    # ---------------------------------------------------------

    def create_run_dir(self) -> Path:
        """
        Erstellt einen neuen Artefakt-Ordner für einen Run.
        Beispiel:
            artifacts/run_2024-01-01_12-30-55/
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self.base_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Artefakt-Ordner erstellt: {run_dir}")
        return run_dir

    # ---------------------------------------------------------
    # MODELLREPRÄSENTATION SPEICHERN
    # ---------------------------------------------------------

    def save_model_repr(self, run_dir: Path, model_repr: str) -> Path:
        """
        Speichert die Modellrepräsentation als Textdatei.
        """
        file_path = run_dir / "model_repr.txt"
        try:
            file_path.write_text(model_repr, encoding="utf-8")
            self.logger.info(f"Modellrepräsentation gespeichert: {file_path}")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Modellrepräsentation: {e}")
        return file_path

    # ---------------------------------------------------------
    # LOG-DATEI SPEICHERN
    # ---------------------------------------------------------

    def save_log(self, run_dir: Path, log_text: str) -> Path:
        """
        Speichert die Log-Ausgabe des Runs.
        """
        file_path = run_dir / "run_log.txt"
        try:
            file_path.write_text(log_text, encoding="utf-8")
            self.logger.info(f"Run-Log gespeichert: {file_path}")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des Logs: {e}")
        return file_path

    # ---------------------------------------------------------
    # DEBUG-DATEI SPEICHERN
    # ---------------------------------------------------------

    def save_debug_info(self, run_dir: Path, debug_text: str) -> Path:
        """
        Speichert Debug-Informationen (stderr, Exceptions etc.).
        """
        file_path = run_dir / "debug_info.txt"
        try:
            file_path.write_text(debug_text, encoding="utf-8")
            self.logger.info(f"Debug-Info gespeichert: {file_path}")
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Debug-Info: {e}")
        return file_path

    # ---------------------------------------------------------
    # ARTEFAKTE KOPIEREN
    # ---------------------------------------------------------

    def copy_artifacts(self, source_dir: str | Path, run_dir: Path):
        """
        Kopiert alle Artefakte aus einem Ordner in den Run-Ordner.
        """
        source_dir = Path(source_dir)
        if not source_dir.exists():
            self.logger.warning(f"Artefakt-Ordner existiert nicht: {source_dir}")
            return

        target_dir = run_dir / "artifacts"
        target_dir.mkdir(exist_ok=True)

        try:
            for item in source_dir.iterdir():
                if item.is_file():
                    shutil.copy(item, target_dir)
                elif item.is_dir():
                    shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True)

            self.logger.info(f"Artefakte kopiert nach: {target_dir}")

        except Exception as e:
            self.logger.error(f"Fehler beim Kopieren der Artefakte: {e}")

    # ---------------------------------------------------------
    # AUFRÄUMEN
    # ---------------------------------------------------------

    def cleanup(self, run_dir: Path):
        """
        Löscht einen Run-Ordner.
        """
        try:
            shutil.rmtree(run_dir)
            self.logger.info(f"Artefakt-Ordner gelöscht: {run_dir}")
        except Exception as e:
            self.logger.error(f"Fehler beim Löschen des Artefakt-Ordners: {e}")
