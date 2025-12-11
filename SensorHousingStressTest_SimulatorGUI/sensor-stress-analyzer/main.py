# main.py

import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    """
    Entry point for Sensor Stress Analyzer application.
    Initializes the Qt application and launches the main window.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Sensor Stress Analyzer v1.0.0")
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
