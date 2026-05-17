# mlflow_local_runner/src/core/runner.py
"""
runner.py – Startet das Nutzer-Skript oder das Template als Subprozess.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

import mlflow

from utils.logger import get_logger
from core.mlflow_client import MLflowClientWrapper


# ---------------------------------------------------------
# TEMPLATE-PFAD DEFINIEREN
# ---------------------------------------------------------
TEMPLATE_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "core" / "script_template.py"
)


class Runner:
    """
    Startet das Nutzer-Skript oder das Template als Subprozess.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.mlflow_client = MLflowClientWrapper()

        # GUI-Callbacks (werden von AppWindow gesetzt)
        self.on_log_info = None
        self.on_log_warning = None
        self.on_log_error = None

    # ---------------------------------------------------------
    # Hilfsfunktionen für GUI-Logging
    # ---------------------------------------------------------

    def _emit_info(self, msg):
        if self.on_log_info:
            self.on_log_info(msg)
        self.logger.info(msg)

    def _emit_warning(self, msg):
        if self.on_log_warning:
            self.on_log_warning(msg)
        self.logger.warning(msg)

    def _emit_error(self, msg):
        if self.on_log_error:
            self.on_log_error(msg)
        self.logger.error(msg)

    # ---------------------------------------------------------
    # RUN STARTEN
    # ---------------------------------------------------------

    def run(self, script_path: str | None, dataset_path: str, mlflow_config: dict) -> dict:
        """
        Führt das Skript aus und gibt ein Ergebnis-Dict zurück.
        """

        # -----------------------------------------------------
        # 1. Skriptpfad bestimmen (Template-Fallback)
        # -----------------------------------------------------
        if not script_path:
            self._emit_info("Kein Nutzer-Skript ausgewählt → Template wird verwendet.")
            script_path = TEMPLATE_SCRIPT_PATH
        else:
            script_path = Path(script_path).resolve()
            self._emit_info(f"Nutzer-Skript wird verwendet: {script_path}")

        script_path = str(Path(script_path).resolve())

        # -----------------------------------------------------
        # 2. MLflow-Konfiguration anwenden
        # -----------------------------------------------------
        tracking_uri = mlflow_config.get("tracking_uri")
        registry_uri = mlflow_config.get("registry_uri")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)

        # Experimentname setzen
        exp_name = mlflow_config.get("experiment_name")
        if exp_name:
            self._emit_info(f"Setze Experiment: {exp_name}")
            mlflow.set_experiment(exp_name)

        # Runname setzen
        run_name = mlflow_config.get("run_name") or None
        if run_name:
            self._emit_info(f"Setze Run-Name: {run_name}")

        # Projektname als Tag
        project_name = mlflow_config.get("project_name")

        # -----------------------------------------------------
        # 3. MLflow-Run starten (nur für Meta-Infos/Tags)
        # -----------------------------------------------------
        with mlflow.start_run(run_name=run_name):
            if project_name:
                mlflow.set_tag("project_name", project_name)

            # -----------------------------------------------------
            # 4. Subprozess starten
            # -----------------------------------------------------
            env = os.environ.copy()
            env["DATASET_PATH"] = dataset_path
            env["MLFLOW_TRACKING_URI"] = tracking_uri
            env["MLFLOW_REGISTRY_URI"] = registry_uri
            env["ARTIFACT_DIR"] = mlflow_config.get("artifact_dir", "")

            env["MODEL_TYPE"] = env.get("MODEL_TYPE", "random_forest")
            env["TUNING"] = env.get("TUNING", "false")
            env["USE_PCA"] = env.get("USE_PCA", "false")

            self._emit_info(f"Starte Subprozess: {script_path}")

            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )

            model_repr_lines = []
            metrics = None
            collecting_model = False

            # -----------------------------------------------------
            # 5. stdout live lesen
            # -----------------------------------------------------
            for raw_line in process.stdout:
                line = raw_line.rstrip("\n")
                self._emit_info(f"[SCRIPT] {line}")

                if line == "MODEL_READY":
                    collecting_model = True
                    continue

                if line == "METRICS_READY":
                    collecting_model = False
                    metrics_raw = next(process.stdout).strip()
                    self._emit_info(f"Metriken empfangen: {metrics_raw}")

                    try:
                        metrics = json.loads(metrics_raw.replace("'", '"'))
                    except Exception:
                        self._emit_error("Fehler beim Parsen der Metriken.")
                        metrics = {}
                    continue

                if collecting_model:
                    model_repr_lines.append(line)

            model_repr = "\n".join(model_repr_lines) if model_repr_lines else None

            # -----------------------------------------------------
            # 6. stderr live lesen
            # -----------------------------------------------------
            stderr_output = process.stderr.read().strip()

            if stderr_output:
                for raw_line in stderr_output.splitlines():
                    line = raw_line.strip()

                    if "WARNING" in line or "Warning" in line:
                        self._emit_warning(f"[SCRIPT-WARNING] {line}")
                        continue

                    if (
                        "ERROR" in line
                        or "Error" in line
                        or "Traceback" in line
                        or "Exception" in line
                    ):
                        self._emit_error(f"[SCRIPT-ERROR] {line}")
                        raise RuntimeError(f"Skriptfehler:\n{stderr_output}")

                    self._emit_warning(f"[SCRIPT-STDERR] {line}")

            process.wait()

            if metrics is None:
                raise RuntimeError("Das Skript hat keine Metriken ausgegeben.")

        # -----------------------------------------------------
        # 7. MLflow-Logging (nachdem der obige Run beendet ist)
        # -----------------------------------------------------
        mlflow_links = self.mlflow_client.log_run(
            metrics=metrics,
            model_repr=model_repr,
            dataset_path=dataset_path,
            artifact_dir=mlflow_config.get("artifact_dir", "")
        )

        # -----------------------------------------------------
        # 8. Ergebnis zurückgeben (erweitert um Projekt/Experiment/Run)
        # -----------------------------------------------------
        return {
            "metrics": metrics,
            "model_repr": model_repr,
            "mlflow_links": mlflow_links,
            "project_name": project_name,
            "experiment_name": exp_name,
            "run_name": run_name,
        }
