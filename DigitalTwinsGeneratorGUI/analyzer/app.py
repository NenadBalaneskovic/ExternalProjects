# analyzer/app.py

import sys
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from analyzer.gui.main_window import MainWindow


def main():
    """
    Entry point for the Telemetry Analyzer (Project B).

    Responsibilities:
        - Initialize Qt application
        - Create and show MainWindow
        - Optionally accept a config path via CLI
    """
    app = QApplication(sys.argv)
    app.setApplicationName("Telemetry Analyzer")
    #app.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    # Optional: config path from CLI, default: ./config.json
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = os.path.join(os.getcwd(), "config.json")

    window = MainWindow(config_path=config_path)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()