# mlflow_local_runner/tests/test_runner.py
"""
test_runner.py – Tests für Runner

Diese Tests mocken:
- subprocess.Popen
- stdout/stderr des Skripts
- MLflowClientWrapper.log_run

Damit wird kein echtes Skript ausgeführt.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from core.runner import Runner


# ---------------------------------------------------------
# HILFSFUNKTION: Fake Subprozess erzeugen
# ---------------------------------------------------------

def make_fake_process(stdout_lines, stderr_text=""):
    """
    Erzeugt ein Fake-Popen-Objekt mit kontrolliertem stdout/stderr.
    """
    process = MagicMock()

    # stdout simulieren (Iterator)
    process.stdout = iter(stdout_lines)

    # stderr simulieren
    process.stderr.read.return_value = stderr_text

    # wait() simulieren
    process.wait.return_value = 0

    return process


# ---------------------------------------------------------
# TEST: Erfolgreicher Run
# ---------------------------------------------------------

@patch("mlflow_local_runner.core.runner.MLflowClientWrapper")
@patch("subprocess.Popen")
def test_runner_success(mock_popen, mock_mlflow_client, tmp_path):
    runner = Runner()

    # Fake stdout mit deinen Markern
    stdout_lines = [
        "MODEL_READY\n",
        "RandomForestClassifier(max_depth=5)\n",
        "METRICS_READY\n",
        "{'accuracy': 0.95, 'f1_score': 0.93}\n",
    ]

    fake_process = make_fake_process(stdout_lines)
    mock_popen.return_value = fake_process

    # Fake MLflow-Links
    mock_mlflow_client.return_value.log_run.return_value = {
        "run_url": "http://localhost/run/1",
        "model_url": "http://localhost/model/1",
        "artifact_url": "http://localhost/artifacts/1",
    }

    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b,c\n1,2,3")

    result = runner.run(
        script_path=None,  # Template-Fallback
        dataset_path=str(dataset),
        mlflow_config={"tracking_uri": "http://localhost"}
    )

    # -----------------------------------------------------
    # PRÜFEN: Modellrepräsentation korrekt
    # -----------------------------------------------------
    assert result["model_repr"] == "RandomForestClassifier(max_depth=5)"

    # -----------------------------------------------------
    # PRÜFEN: Metriken korrekt geparst
    # -----------------------------------------------------
    assert result["metrics"]["accuracy"] == 0.95
    assert result["metrics"]["f1_score"] == 0.93

    # -----------------------------------------------------
    # PRÜFEN: MLflow-Links korrekt übernommen
    # -----------------------------------------------------
    assert "run_url" in result
    assert "model_url" in result
    assert "artifact_url" in result


# ---------------------------------------------------------
# TEST: Skript gibt keine Metriken aus
# ---------------------------------------------------------

@patch("subprocess.Popen")
def test_runner_missing_metrics(mock_popen, tmp_path):
    runner = Runner()

    stdout_lines = [
        "MODEL_READY\n",
        "SomeModel()\n",
        # METRICS_READY fehlt absichtlich
    ]

    fake_process = make_fake_process(stdout_lines)
    mock_popen.return_value = fake_process

    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b,c\n1,2,3")

    with pytest.raises(RuntimeError):
        runner.run(
            script_path=None,
            dataset_path=str(dataset),
            mlflow_config={}
        )


# ---------------------------------------------------------
# TEST: Skript wirft Fehler (stderr)
# ---------------------------------------------------------

@patch("subprocess.Popen")
def test_runner_script_error(mock_popen, tmp_path):
    runner = Runner()

    stdout_lines = []
    stderr_text = "Traceback: Something went wrong"

    fake_process = make_fake_process(stdout_lines, stderr_text=stderr_text)
    mock_popen.return_value = fake_process

    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b,c\n1,2,3")

    with pytest.raises(RuntimeError):
        runner.run(
            script_path=None,
            dataset_path=str(dataset),
            mlflow_config={}
        )
