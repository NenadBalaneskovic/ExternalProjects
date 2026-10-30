# mlflow_local_runner/src/core/mlflow_client.py
"""
mlflow_client.py – Wrapper für MLflow-Operationen (tokenfrei)

Dieses Modul übernimmt:
- Starten eines MLflow-Runs
- Loggen von Metriken
- Loggen des Input-Datasets
- Loggen von Artefakten
- Loggen der Modellrepräsentation (das echte Modell wird im Template geloggt!)
- Zurückgeben von MLflow-Links für die GUI
"""

import os
from pathlib import Path

import mlflow

from utils.logger import get_logger


class MLflowClientWrapper:

    def __init__(self):
        self.logger = get_logger(__name__)

    # ---------------------------------------------------------
    # RUN LOGGEN
    # ---------------------------------------------------------

    def log_run(
        self,
        metrics: dict,
        model_repr: str,
        dataset_path: str,
        artifact_dir: str,
        model=None,   # bleibt für Rückwärtskompatibilität, wird aber NICHT genutzt
    ) -> dict:
        """
        Führt einen vollständigen MLflow-Run durch und gibt Links zurück.
        """

        # tracking_uri darf NIEMALS leer sein → sonst entstehen relative Links
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "http://127.0.0.1:5000"
        registry_uri = os.environ.get("MLFLOW_REGISTRY_URI") or tracking_uri

        self.logger.info(f"MLflow Tracking URI: {tracking_uri}")
        self.logger.info(f"MLflow Registry URI: {registry_uri}")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)

        # -----------------------------------------------------
        # 1. Run starten
        # -----------------------------------------------------
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            exp_id = run.info.experiment_id
            self.logger.info(f"MLflow-Run gestartet: {run_id}")

            # -------------------------------------------------
            # 2. Dataset loggen
            # -------------------------------------------------
            try:
                mlflow.log_artifact(dataset_path, artifact_path="input_dataset")
                self.logger.info("Dataset erfolgreich geloggt.")
            except Exception as e:
                self.logger.error(f"Dataset konnte nicht geloggt werden: {e}")

            # -------------------------------------------------
            # 3. Metriken loggen
            # -------------------------------------------------
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            self.logger.info("Metriken erfolgreich geloggt.")

            # -------------------------------------------------
            # 4. Modellrepräsentation loggen (Text)
            # -------------------------------------------------
            model_txt_path = Path("model_repr.txt")
            try:
                model_txt_path.write_text(model_repr)
                mlflow.log_artifact(str(model_txt_path), artifact_path="model_info")
                self.logger.info("Modellrepräsentation erfolgreich geloggt.")
            except Exception as e:
                self.logger.error(f"Modellrepräsentation konnte nicht gespeichert werden: {e}")

            # -------------------------------------------------
            # 5. Artefakte loggen
            # -------------------------------------------------
            if artifact_dir and Path(artifact_dir).exists():
                try:
                    mlflow.log_artifacts(artifact_dir, artifact_path="artifacts")
                    self.logger.info("Artefakte erfolgreich geloggt.")
                except Exception as e:
                    self.logger.error(f"Artefakte konnten nicht geloggt werden: {e}")

        # -----------------------------------------------------
        # 6. Modell-URL (Modell wurde im Template registriert!)
        # -----------------------------------------------------
        model_url = f"{tracking_uri}/#/models/local_runner_model"

        # -----------------------------------------------------
        # 7. MLflow-Links (MLflow 3.x-kompatibel)
        # -----------------------------------------------------
        run_url = f"{tracking_uri}/#/experiments/{exp_id}/runs/{run_id}"
        artifact_url = f"{tracking_uri}/#/experiments/{exp_id}/runs/{run_id}/artifacts"

        return {
            "run_url": run_url,
            "artifact_url": artifact_url,
            "model_url": model_url,
            "run_id": run_id,
            "experiment_id": exp_id,
            "tracking_uri": tracking_uri,
        }
