"""
components.py
-------------
Reusable GUI components for the Slurm HPC–QPU Workflow Orchestrator.

This module provides:
    - upload section
    - analysis panel
    - Slurm preview panel
    - credentials section
    - generate button

All components are pure factories:
    - no event loop
    - no workflow execution
    - safe for testing
"""

import PySimpleGUI as sg


# ----------------------------------------------------------------------
# Upload Section
# ----------------------------------------------------------------------

def upload_section():
    """
    Component: Workflow upload section.
    """
    return [
        [
            sg.Text("Select Python Workflow:", font=("Arial", 12)),
            sg.Input(key="WORKFLOW_PATH", enable_events=True),
            sg.FileBrowse(button_text="Browse"),
            sg.Button("Upload", key="UPLOAD_BUTTON")
        ]
    ]


# ----------------------------------------------------------------------
# Workflow Analysis Panel
# ----------------------------------------------------------------------

def analysis_panel():
    """
    Component: Workflow analysis multiline panel.
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


# ----------------------------------------------------------------------
# Slurm Preview Panel
# ----------------------------------------------------------------------

def slurm_preview_panel():
    """
    Component: Slurm script preview multiline panel.
    """
    return sg.Multiline(
        key="SLURM_PREVIEW",
        size=(80, 20),
        disabled=True,
        autoscroll=True
    )


# ----------------------------------------------------------------------
# Credentials Section
# ----------------------------------------------------------------------

def credentials_toggle():
    """
    Component: Checkbox that enables/disables manual QPU credentials.
    """
    return sg.Checkbox(
        "Enable manual QPU credentials",
        key="ENABLE_CREDS",
        enable_events=True
    )


def credentials_section():
    """
    Component: Manual QPU credentials input fields.
    Initially hidden.
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


# ----------------------------------------------------------------------
# Generate Slurm Button
# ----------------------------------------------------------------------

def generate_button():
    """
    Component: Button to generate Slurm script.
    """
    return sg.Button(
        "Generate Slurm Script",
        key="GENERATE_SLURM",
        button_color=("white", "green")
    )