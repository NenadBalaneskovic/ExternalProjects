"""
credentials_section.py
----------------------
GUI layout component for the manual QPU credentials section.

This module provides:
    - a pure layout factory for the credentials panel
    - no event loop
    - no workflow execution
    - safe for testing

It is used by main_layout.py and main_gui.py.
"""

import PySimpleGUI as sg


def build_credentials_section():
    """
    Build the credentials section layout.
    Initially hidden; visibility controlled by ENABLE_CREDS checkbox.
    """

    return sg.Column(
        [
            [sg.Text("QPU Credentials", font=("Arial", 12, "bold"))],
            [sg.Text("API Key:"), sg.Input(key="API_KEY")],
            [sg.Text("Runtime URL:"), sg.Input(key="RUNTIME_URL")],
        ],
        key="CREDS_SECTION",
        visible=False
    )