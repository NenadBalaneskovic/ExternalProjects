"""
workflow_analysis_panel.py
--------------------------
GUI layout component for the workflow analysis panel.

This module provides:
    - a pure layout factory for the analysis multiline element
    - no event loop
    - no workflow execution
    - safe for testing

It is used by main_layout.py and main_gui.py.
"""

import PySimpleGUI as sg


def build_workflow_analysis_panel():
    """
    Build the workflow analysis panel layout.
    This panel displays static AST + classification results.
    """

    return sg.Column(
        [
            [sg.Text("Workflow Analysis", font=("Arial", 14, "bold"))],
            [
                sg.Multiline(
                    key="ANALYSIS_PANEL",
                    size=(80, 15),
                    disabled=True,
                    autoscroll=True
                )
            ]
        ],
        key="ANALYSIS_PANEL"
    )