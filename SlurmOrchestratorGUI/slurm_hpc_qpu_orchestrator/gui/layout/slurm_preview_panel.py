"""
slurm_preview_panel.py
----------------------
GUI layout component for the Slurm script preview panel.

This module provides:
    - a pure layout factory for the Slurm preview multiline element
    - no event loop
    - no workflow execution
    - safe for testing

It is used by main_layout.py and main_gui.py.
"""

import PySimpleGUI as sg


def build_slurm_preview_panel():
    """
    Build the Slurm preview panel layout.
    This panel displays the generated Slurm script text.
    """

    return sg.Multiline(
        key="SLURM_PREVIEW",
        size=(80, 20),
        disabled=True,
        autoscroll=True
    )