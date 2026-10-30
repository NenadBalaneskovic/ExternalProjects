# mlflow_local_runner/tests/test_mlflow_client.py
"""
test_mlflow_client.py – Tests für MLflowClientWrapper

Diese Tests mocken MLflow vollständig, damit:
- keine echten Runs erzeugt werden
- keine echten Artefakte geschrieben werden
- keine Registry benötigt wird
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from core.mlflow_client import MLflowClientWrapper


@pytest.fixture
def mlflow_env(monkeypatch, tmp_path):
    """Setzt MLflow-ENV-Variablen für den Test."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "http://localhost:5000")
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def client():
    return MLflowClientWrapper()


# ---------------------------------------------------------
# MOCK-HILFSFUNKTIONEN
# ---------------------------------------------------------

def mock_mlflow_run():
    """Erzeugt ein Fake-Run-Objekt mit Run-ID und Experiment-ID."""
    run = MagicMock()
    run.info.run_id = "12345"
    run.info.experiment_id = "7"
    return run


# ---------------------------------------------------------
# TESTS
# ---------------------------------------------------------

@patch("mlflow.register_model")
@patch("mlflow.log_artifacts")
@patch("mlflow.log_artifact")
@patch("mlflow.log_metric")
@patch("mlflow.start_run")
def test_log_run_success(
    mock_start_run,
    mock_log_metric,
    mock_log_artifact,
    mock_log_artifacts,
    mock_register_model,
    client,
    mlflow_env,
):
    """Testet, ob log_run() MLflow korrekt aufruft."""

    # Fake-Run zurückgeben
    mock_start_run.return_value.__enter__.return_value = mock_mlflow_run()

    metrics = {"accuracy": 0.95, "f1_score": 0.93}
    model_repr = "RandomForestClassifier()"
    dataset_path = str(mlflow_env / "data.csv")
    (mlflow_env / "data.csv").write_text("a,b,c\n1,2,3")

    result = client.log_run(
        metrics=metrics,
        model_repr=model_repr,
        dataset_path=dataset_path,
        artifact_dir=str(mlflow_env)
    )

    # -----------------------------------------------------
    # PRÜFEN: Metriken wurden geloggt
    # -----------------------------------------------------
    assert mock_log_metric.call_count == 2
    mock_log_metric.assert_any_call("accuracy", 0.95)
    mock_log_metric.assert_any_call("f1_score", 0.93)

    # -----------------------------------------------------
    # PRÜFEN: Dataset wurde geloggt
    # -----------------------------------------------------
    mock_log_artifact.assert_called()

    # -----------------------------------------------------
    # PRÜFEN: Artefakte wurden geloggt
    # -----------------------------------------------------
    mock_log_artifacts.assert_called()

    # -----------------------------------------------------
    # PRÜFEN: Modellregistrierung wurde versucht
    # -----------------------------------------------------
    mock_register_model.assert_called()

    # -----------------------------------------------------
    # PRÜFEN: Rückgabeformat korrekt
    # -----------------------------------------------------
    assert "run_url" in result
    assert "model_url" in result
    assert "artifact_url" in result

    assert result["run_url"] == "http://localhost:5000/#/experiments/7/runs/12345"
    assert result["artifact_url"] == "http://localhost:5000/#/experiments/7/runs/12345/artifacts"
    assert result["model_url"] == "http://localhost:5000/#/models/local_runner_model"
