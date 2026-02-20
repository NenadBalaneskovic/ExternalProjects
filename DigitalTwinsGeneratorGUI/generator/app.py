# generator/app.py

import sys
from PySide6.QtWidgets import QApplication
from generator.gui.main_window import MainWindow


def main():
    """
    Entry point for the Telemetry Generator GUI.

    Responsibilities:
        - Create the Qt application
        - Instantiate MainWindow
        - Show the GUI
        - Start the Qt event loop
    """
    import os
    print("GENERATOR GUI PID:", os.getpid())
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()