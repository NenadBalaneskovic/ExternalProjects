# gui/main_window.py

from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QScrollArea
from gui.sidebar_controls import Sidebar
from gui.visualization_panel import VisualizationPanel
from gui.event_handlers import EventHandlers


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sensor Stress Analyzer v1.0.0")
        self.resize(1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # Sidebar controls (left)
        self.sidebar = Sidebar()
        layout.addWidget(self.sidebar)

        # Visualization panel wrapped in scroll area (right)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        self.visualization = VisualizationPanel()
        scroll_area.setWidget(self.visualization)

        layout.addWidget(scroll_area)

        # Event handlers wiring
        self.handlers = EventHandlers(self.sidebar, self.visualization)
