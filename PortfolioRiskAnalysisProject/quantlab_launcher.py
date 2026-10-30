from PyQt5.QtWidgets import QApplication
from main_gui import QuantCanvas

if __name__ == "__main__":
    print("Launching QuantCanvas...")
    app = QApplication([])
    window = QuantCanvas()
    window.show()
    app.exec_()
