from PyQt5.QtWidgets import QApplication
from gui import InvoiceTrackerApp
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InvoiceTrackerApp()
    window.show()
    sys.exit(app.exec_())
