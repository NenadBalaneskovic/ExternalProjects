"""
gui.layout
----------
Public layout constructors for the Slurm HPC–QPU Workflow Orchestrator GUI.

This package exposes:
    - build_main_layout(): top-level layout builder used by main_gui.py

Internal layout fragments (panels, frames, sections) remain private.
"""

from .main_layout import build_main_layout

__all__ = [
    "build_main_layout",
]
