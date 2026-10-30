"""
layout.py

Provides reusable, theme-aware GUI layout components for the MemAlloc Optimizer.
This module keeps app.py clean by centralizing all PySimpleGUI layout definitions.
"""

import PySimpleGUI as sg
from typing import List


# ============================================================
# Section Builders
# ============================================================

def section_header(text: str) -> List:
    """Consistent section header styling."""
    return [sg.Text(text, font=("Segoe UI", 12, "bold"))]


def script_loader_section() -> List:
    return [
        *section_header("Upload Python Script"),
        [
            sg.Input(key="-SCRIPT_PATH-", enable_events=True),
            sg.FileBrowse(file_types=(("Python Files", "*.py"),)),
            sg.Button("Load Script"),
        ],
        [sg.HorizontalSeparator()],
    ]


def analysis_section() -> List:
    return [
        *section_header("Static Analysis"),
        [
            sg.Button("Run Analysis"),
        ],
        [
            sg.Multiline(
                key="-ANALYSIS_OUT-",
                size=(80, 10),
                autoscroll=True,
                font=("Consolas", 10),
            )
        ],
        [sg.HorizontalSeparator()],
    ]


def strategy_section() -> List:
    return [
        *section_header("Optimization Strategies"),
        [
            sg.Checkbox("Cython Memoryviews", key="-CYTHON-"),
            sg.Checkbox("Numba JIT", key="-NUMBA-"),
            sg.Checkbox("Preallocate Buffers", key="-PREALLOC-"),
            sg.Checkbox("Optimize Layout", key="-LAYOUT-"),
        ],
        [
            sg.Button("Build Optimization Plan"),
        ],
        [
            sg.Multiline(
                key="-PLAN_OUT-",
                size=(80, 10),
                autoscroll=True,
                font=("Consolas", 10),
            )
        ],
        [sg.HorizontalSeparator()],
    ]


def codegen_section() -> List:
    return [
        *section_header("Code Generation"),
        [
            sg.Button("Generate Code"),
        ],
        [
            sg.Multiline(
                key="-CODEGEN_OUT-",
                size=(80, 10),
                autoscroll=True,
                font=("Consolas", 10),
            )
        ],
        [sg.HorizontalSeparator()],
    ]


def execution_section() -> List:
    return [
        *section_header("Execution"),
        [
            sg.Button("Run Baseline"),
            sg.Button("Run Optimized"),
        ],
        [
            sg.Multiline(
                key="-EXEC_OUT-",
                size=(80, 10),
                autoscroll=True,
                font=("Consolas", 10),
            )
        ],
        [sg.HorizontalSeparator()],
    ]


def plots_section() -> List:
    """
    ⭐ This section was originally missing in our GUI.
    ⭐ Without it, the plot generator is NEVER called.
    ⭐ This is why no plots ever appeared.
    """
    return [
        *section_header("Plots"),
        [
            sg.Button("Generate Plots", key="-GENERATE_PLOTS-"),
        ],
        [
            sg.Image(key="-PLOT_IMG-", size=(600, 300)),
        ],
        [sg.HorizontalSeparator()],
    ]


# ============================================================
# Main Layout Builder
# ============================================================

def build_main_layout() -> List:
    """
    Returns the full window layout as a list of sections.
    app.py simply calls this function.
    """
    return (
        script_loader_section()
        + analysis_section()
        + strategy_section()
        + codegen_section()
        + execution_section()
        + plots_section()   # ⭐ Now included
    )
