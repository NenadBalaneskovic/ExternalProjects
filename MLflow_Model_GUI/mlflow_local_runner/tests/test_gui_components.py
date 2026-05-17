# mlflow_local_runner/tests/test_gui_components.py
"""
test_gui_components.py – GUI-Tests für MLflow Local Runner

Diese Tests verwenden pytest + pytest-qt.
Sie prüfen:
- Initialisierung der Panels
- Signalfluss
- UI-Interaktionen
- Validierungslogik
"""

import pytest
from PySide6.QtCore import Qt

from gui.app_window import AppWindow
from gui.upload_panel import UploadPanel
from gui.config_panel import ConfigPanel
from gui.run_panel import RunPanel
from gui.results_panel import ResultsPanel


# ---------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------

@pytest.fixture
def app_window(qtbot):
    """Erstellt ein AppWindow für Tests."""
    window = AppWindow(config={})
    qtbot.addWidget(window)
    window.show()
    return window


@pytest.fixture
def upload_panel(qtbot):
    panel = UploadPanel()
    qtbot.addWidget(panel)
    panel.show()
    return panel


@pytest.fixture
def config_panel(qtbot):
    panel = ConfigPanel(initial_config={})
    qtbot.addWidget(panel)
    panel.show()
    return panel


@pytest.fixture
def run_panel(qtbot):
    panel = RunPanel()
    qtbot.addWidget(panel)
    panel.show()
    return panel


@pytest.fixture
def results_panel(qtbot):
    panel = ResultsPanel()
    qtbot.addWidget(panel)
    panel.show()
    return panel


# ---------------------------------------------------------
# UPLOAD PANEL TESTS
# ---------------------------------------------------------

def test_upload_panel_initial_state(upload_panel):
    assert upload_panel.script_input.text() == ""
    assert upload_panel.dataset_input.text() == ""


def test_upload_panel_script_signal(upload_panel, qtbot, tmp_path):
    test_file = tmp_path / "test_script.py"
    test_file.write_text("print('hello')")

    with qtbot.waitSignal(upload_panel.script_selected, timeout=500):
        upload_panel._select_script = lambda: upload_panel.script_selected.emit(str(test_file))
        upload_panel._select_script()

    assert upload_panel.script_path == str(test_file)


def test_upload_panel_dataset_signal(upload_panel, qtbot, tmp_path):
    test_file = tmp_path / "data.csv"
    test_file.write_text("a,b,c\n1,2,3")

    with qtbot.waitSignal(upload_panel.dataset_selected, timeout=500):
        upload_panel._select_dataset = lambda: upload_panel.dataset_selected.emit(str(test_file))
        upload_panel._select_dataset()

    assert upload_panel.dataset_path == str(test_file)


# ---------------------------------------------------------
# CONFIG PANEL TESTS
# ---------------------------------------------------------

def test_config_panel_loads_defaults(config_panel):
    cfg = config_panel.get_config()
    assert "tracking_uri" in cfg
    assert "registry_uri" in cfg
    assert "artifact_dir" in cfg


def test_config_panel_save_signal(config_panel, qtbot):
    with qtbot.waitSignal(config_panel.config_saved, timeout=500):
        config_panel._save_config()


# ---------------------------------------------------------
# RUN PANEL TESTS
# ---------------------------------------------------------

def test_run_panel_button_disabled(run_panel):
    run_panel.set_running(True)
    assert not run_panel.btn_start.isEnabled()

    run_panel.set_running(False)
    assert run_panel.btn_start.isEnabled()


def test_run_panel_log_append(run_panel):
    run_panel.append_log("Test log")
    assert "Test log" in run_panel.log_output.toPlainText()


# ---------------------------------------------------------
# RESULTS PANEL TESTS
# ---------------------------------------------------------

def test_results_panel_display(results_panel):
    results = {
        "metrics": {"accuracy": 0.95},
        "model_repr": "RandomForestClassifier()",
        "mlflow_links": {"run_url": "http://localhost/run/1"}
    }

    results_panel.display_results(results)

    assert "0.95" in results_panel.metrics_text.toPlainText()
    assert "RandomForestClassifier" in results_panel.model_text.toPlainText()
    assert "http://localhost" in results_panel.links_text.toPlainText()


# ---------------------------------------------------------
# APP WINDOW TESTS
# ---------------------------------------------------------

def test_app_window_initial_state(app_window):
    assert app_window.script_path is None
    assert app_window.dataset_path is None


def test_app_window_receives_script_signal(app_window):
    app_window._on_script_selected("test.py")
    assert app_window.script_path == "test.py"


def test_app_window_receives_dataset_signal(app_window):
    app_window._on_dataset_selected("data.csv")
    assert app_window.dataset_path == "data.csv"
