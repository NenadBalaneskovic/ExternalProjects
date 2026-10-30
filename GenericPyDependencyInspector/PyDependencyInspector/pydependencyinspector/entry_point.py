"""
pydependencyinspector/entry_point.py

Application entry point for PyDependencyInspector.

Responsibilities:
- Initialize logging.
- Create a QApplication (safe for Jupyter/VSCode).
- Launch the MainWindow.
- Provide a clean main() function for PyInstaller and CLI execution.

This module is intentionally:
- Minimal.
- Deterministic.
- PyInstaller-friendly.
"""

from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication

from .gui.main_window import MainWindow, ProjectState
from .utils.logging_utils import configure_logging, get_logger


def main() -> None:
    """
    Launch the PyDependencyInspector GUI.
    Safe to call from PyInstaller, CLI, or python -m.
    """
    # Initialize logging (GUI callback is attached by LogPanel)
    configure_logging()
    logger = get_logger(__name__)
    logger.info("Starting PyDependencyInspector…")

    # Safe QApplication handling
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Shared project state
    state = ProjectState()

    # Main window
    window = MainWindow(project_state=state)
    window.show()

    logger.info("GUI initialized successfully.")
    app.exec()


if __name__ == "__main__":
    main()
