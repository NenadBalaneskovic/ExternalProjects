# tests/test_gui_interactions.py

import pytest
from gui.sidebar_controls import SidebarControls
from gui.visualization_panel import VisualizationPanel
from gui.event_handlers import EventHandlers


def test_gui_event_handler_runs(qtbot):
    sidebar = SidebarControls()
    visualization = VisualizationPanel()
    handlers = EventHandlers(sidebar, visualization)

    # Simulate button click
    qtbot.mouseClick(sidebar.run_button, qtbot.LeftButton)

    # Ensure visualization labels update
    assert "Rod structure" in visualization.rod_label.text() or "FEM" in visualization.fem_label.text()