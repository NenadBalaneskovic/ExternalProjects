"""
main.py — Entry point for the Data Privacy Workbench GUI

This script initializes the PySide6 application, sets up logging,
loads configuration, and launches the main window that hosts all tabs:
Data Cleaning, Anonymization, Pseudonymization, Synthetic Data, and Logs.

Author: Nenad
Date: May 2026
"""

import sys
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QIcon

# Local imports
from gui.app_window import AppWindow
from core.logger import get_logger
from core.config import ConfigLoader


def main() -> None:
    """
    Main entry point for the Data Privacy Workbench application.

    Responsibilities:
    -----------------
    1. Initialize logging system.
    2. Load user configuration (paths, API key, preferences).
    3. Create and start the Qt application.
    4. Instantiate and show the main window.
    5. Handle startup and shutdown gracefully.
    """

    # --- 1. Initialize logging ---
    logger = get_logger("DataPrivacyWorkbench")
    logger.info("Starting Data Privacy Workbench GUI...")

    # --- 2. Load configuration ---
    try:
        config = ConfigLoader().load()
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        config = {}
        QMessageBox.warning(
            None,
            "Configuration Error",
            f"Could not load configuration.\n\nDetails:\n{e}",
        )

    # --- 3. Initialize Qt application ---
    app = QApplication(sys.argv)
    app.setApplicationName("Data Privacy Workbench")
    app.setWindowIcon(QIcon("assets/icons/settings.png"))

    # --- 4. Create and show main window ---
    try:
        window = AppWindow(config=config, logger=logger)
        window.show()
        logger.info("Main window displayed successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize main window: {e}")
        QMessageBox.critical(
            None,
            "Startup Error",
            f"Critical error while initializing GUI.\n\nDetails:\n{e}",
        )
        sys.exit(1)

    # --- 5. Start event loop ---
    try:
        exit_code = app.exec()
        logger.info("Application closed gracefully.")
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(f"Unhandled exception in event loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
