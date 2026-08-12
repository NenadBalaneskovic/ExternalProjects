"""
GUI package for the Slurm HPC–QPU Workflow Orchestrator.

This module exposes the public GUI entry points:
    - build_main_window(): construct the full GUI layout
    - run_gui(): start the GUI event loop

Internal modules (layout, dialogs, panels) remain private.
"""

def build_main_window():
    from .main_gui import build_main_window
    return build_main_window()

def run_gui():
    from .main_gui import run_gui
    return run_gui()

__all__ = [
    "build_main_window",
    "run_gui",
]
