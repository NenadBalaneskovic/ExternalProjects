"""
file_dialogs.py
---------------
Utility functions for safe file and directory selection in the
Slurm HPC–QPU Workflow Orchestrator GUI.

This module:
    - wraps PySimpleGUI dialogs
    - validates paths
    - provides safe, testable abstractions
    - NEVER executes user workflow code
"""

import PySimpleGUI as sg
from pathlib import Path


# ----------------------------------------------------------------------
# Workflow File Selection
# ----------------------------------------------------------------------

def select_workflow_file() -> Path | None:
    """
    Open a file dialog for selecting a Python workflow file.
    Returns:
        Path object if a valid file is selected,
        None otherwise.
    """
    file_path = sg.popup_get_file(
        "Select Python Workflow",
        file_types=(("Python Files", "*.py"),),
        no_window=True
    )

    if not file_path:
        return None

    p = Path(file_path)
    return p if p.exists() and p.suffix == ".py" else None


# ----------------------------------------------------------------------
# Slurm Script Save Dialog
# ----------------------------------------------------------------------

def select_slurm_save_path(default_name: str = "job.slurm") -> Path | None:
    """
    Open a save-as dialog for writing a Slurm script.
    Returns:
        Path object if user selects a valid save location,
        None otherwise.
    """
    file_path = sg.popup_get_file(
        "Save Slurm Script As",
        save_as=True,
        default_extension=".slurm",
        initial_file=default_name,
        no_window=True
    )

    if not file_path:
        return None

    p = Path(file_path)
    return p if p.suffix == ".slurm" else p.with_suffix(".slurm")


# ----------------------------------------------------------------------
# Directory Selection
# ----------------------------------------------------------------------

def select_output_directory() -> Path | None:
    """
    Open a directory selection dialog.
    Returns:
        Path object if a valid directory is selected,
        None otherwise.
    """
    dir_path = sg.popup_get_folder(
        "Select Output Directory",
        no_window=True
    )

    if not dir_path:
        return None

    p = Path(dir_path)
    return p if p.exists() and p.is_dir() else None


# ----------------------------------------------------------------------
# Path Validation Helpers
# ----------------------------------------------------------------------

def validate_python_file(path: str | Path) -> bool:
    """
    Validate that a given path points to an existing Python file.
    """
    p = Path(path)
    return p.exists() and p.is_file() and p.suffix == ".py"


def validate_directory(path: str | Path) -> bool:
    """
    Validate that a given path points to an existing directory.
    """
    p = Path(path)
    return p.exists() and p.is_dir()