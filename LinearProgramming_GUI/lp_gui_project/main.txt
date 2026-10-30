from PyQt5.QtWidgets import QApplication
from gui import OptimizationApp
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OptimizationApp()
    window.show()
    sys.exit(app.exec_())