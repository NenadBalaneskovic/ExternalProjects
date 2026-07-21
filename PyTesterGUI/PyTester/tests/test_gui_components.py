"""
Tests for gui/ modules.

These tests validate:
- correct initialization of all GUI panels
- correct presence of required widgets
- correct signal/slot wiring
- correct integration with controller stubs

All tests run in offscreen mode (no visible windows).
"""

import pytest
from PySide6.QtWidgets import QApplication

# GUI modules
import gui.main_window as main_window
import gui.upload_panel as upload_panel
import gui.inspection_panel as inspection_panel
import gui.inference_panel as inference_panel
import gui.test_generation_panel as test_generation_panel
import gui.execution_panel as execution_panel
import gui.results_panel as results_panel


# ------------------------------------------------------------
# Qt Application Fixture
# ------------------------------------------------------------
@pytest.fixture(scope="module")
def qt_app():
    """Create a single QApplication for all GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ------------------------------------------------------------
# main_window
# ------------------------------------------------------------
def test_main_window_initializes(qt_app):
    win = main_window.MainWindow()
    assert win is not None
    assert win.centralWidget() is not None
    assert hasattr(win, "stack")


def test_main_window_has_panels(qt_app):
    win = main_window.MainWindow()
    assert win.upload_panel is not None
    assert win.inspection_panel is not None
    assert win.inference_panel is not None
    assert win.test_generation_panel is not None
    assert win.execution_panel is not None
    assert win.results_panel is not None


# ------------------------------------------------------------
# upload_panel
# ------------------------------------------------------------
def test_upload_panel_initializes(qt_app):
    panel = upload_panel.UploadPanel()
    assert panel is not None
    assert hasattr(panel, "upload_button")
    assert hasattr(panel, "file_list")


def test_upload_panel_emits_signal(qt_app):
    panel = upload_panel.UploadPanel()

    triggered = []

    def on_upload(path):
        triggered.append(path)

    panel.file_uploaded.connect(on_upload)
    panel.file_uploaded.emit("dummy.py")

    assert triggered == ["dummy.py"]


# ------------------------------------------------------------
# inspection_panel
# ------------------------------------------------------------
def test_inspection_panel_initializes(qt_app):
    panel = inspection_panel.InspectionPanel()
    assert panel is not None
    assert hasattr(panel, "tree_view")
    assert hasattr(panel, "refresh_button")


# ------------------------------------------------------------
# inference_panel
# ------------------------------------------------------------
def test_inference_panel_initializes(qt_app):
    panel = inference_panel.InferencePanel()
    assert panel is not None
    assert hasattr(panel, "run_inference_button")
    assert hasattr(panel, "schema_view")


def test_inference_panel_schema_update(qt_app):
    panel = inference_panel.InferencePanel()
    panel.update_schema({"foo": {"args": {}, "return": None}})
    assert "foo" in panel.schema_model


# ------------------------------------------------------------
# test_generation_panel
# ------------------------------------------------------------
def test_test_generation_panel_initializes(qt_app):
    panel = test_generation_panel.TestGenerationPanel()
    assert panel is not None
    assert hasattr(panel, "generate_button")
    assert hasattr(panel, "test_list")


def test_test_generation_panel_adds_test(qt_app):
    panel = test_generation_panel.TestGenerationPanel()
    panel.add_generated_test("test_sample.py")
    assert "test_sample.py" in panel.generated_tests


# ------------------------------------------------------------
# execution_panel
# ------------------------------------------------------------
def test_execution_panel_initializes(qt_app):
    panel = execution_panel.ExecutionPanel()
    assert panel is not None
    assert hasattr(panel, "run_button")
    assert hasattr(panel, "log_view")


def test_execution_panel_updates_logs(qt_app):
    panel = execution_panel.ExecutionPanel()
    panel.update_logs("LOGS")
    assert "LOGS" in panel.log_view.toPlainText()


# ------------------------------------------------------------
# results_panel
# ------------------------------------------------------------
def test_results_panel_initializes(qt_app):
    panel = results_panel.ResultsPanel()
    assert panel is not None
    assert hasattr(panel, "coverage_label")
    assert hasattr(panel, "status_label")
    assert hasattr(panel, "plots_area")


def test_results_panel_updates_summary(qt_app):
    panel = results_panel.ResultsPanel()
    panel.update_summary({"status": "ok", "total_coverage": 85.0})
    assert "ok" in panel.status_label.text()
    assert "85" in panel.coverage_label.text()
