# mlflow_local_runner/tests/test_gui_end_to_end.py
"""
test_gui_end_to_end.py – End-to-End GUI Test für MLflow Local Runner

Dieser Test simuliert:
- Auswahl von Script & Dataset
- Laden der Config
- Start eines Runs
- Empfang der Runner-Ergebnisse
- Anzeige im ResultsPanel

Runner + MLflow werden vollständig gemockt.
"""

import pytest
from unittest.mock import patch, MagicMock

from gui.app_window import AppWindow


@pytest.fixture
def app(qtbot):
    """Erstellt das Hauptfenster."""
    window = AppWindow(config={})
    qtbot.addWidget(window)
    window.show()
    return window


# ---------------------------------------------------------
# END-TO-END TEST
# ---------------------------------------------------------

@patch("mlflow_local_runner.gui.app_window.Runner")
def test_gui_end_to_end(mock_runner_class, qtbot, tmp_path, app):
    """
    Simuliert einen kompletten Run:
    - Script auswählen
    - Dataset auswählen
    - Run starten
    - Ergebnisse anzeigen
    """

    # -----------------------------------------------------
    # 1. Fake Runner konfigurieren
    # -----------------------------------------------------
    mock_runner = MagicMock()
    mock_runner_class.return_value = mock_runner

    mock_runner.run.return_value = {
        "metrics": {"accuracy": 0.97},
        "model_repr": "RandomForestClassifier(max_depth=5)",
        "mlflow_links": {"run_url": "http://localhost/run/1"}
    }

    # -----------------------------------------------------
    # 2. Fake Script & Dataset erzeugen
    # -----------------------------------------------------
    script = tmp_path / "script.py"
    script.write_text("print('hello')")

    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b,c\n1,2,3")

    # -----------------------------------------------------
    # 3. Script auswählen
    # -----------------------------------------------------
    app._on_script_selected(str(script))
    assert app.script_path == str(script)

    # -----------------------------------------------------
    # 4. Dataset auswählen
    # -----------------------------------------------------
    app._on_dataset_selected(str(dataset))
    assert app.dataset_path == str(dataset)

    # -----------------------------------------------------
    # 5. Run starten
    # -----------------------------------------------------
    qtbot.mouseClick(app.run_panel.btn_start, qtbot.MouseButton.LeftButton)

    # Runner muss aufgerufen worden sein
    mock_runner.run.assert_called_once()

    # -----------------------------------------------------
    # 6. Ergebnisse prüfen
    # -----------------------------------------------------
    results_text = app.results_panel.metrics_text.toPlainText()
    model_text = app.results_panel.model_text.toPlainText()
    links_text = app.results_panel.links_text.toPlainText()

    assert "0.97" in results_text
    assert "RandomForestClassifier" in model_text
    assert "http://localhost/run/1" in links_text
