# gui/sidebar_controls.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QCheckBox, QPushButton, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class Sidebar(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setSpacing(12)
        self.setLayout(layout)

        font_label = QFont("Arial", 11)

        # n-gon slider
        self.n_label = QLabel("n-gon: 12 corners")
        self.n_label.setFont(font_label)
        self.n_slider = QSlider(Qt.Horizontal)
        self.n_slider.setMinimum(3)
        self.n_slider.setMaximum(20)
        self.n_slider.setValue(12)
        self.n_slider.setTickInterval(1)
        self.n_slider.setTickPosition(QSlider.TicksBelow)
        self.n_slider.valueChanged.connect(self._update_n_label)
        layout.addWidget(self.n_label)
        layout.addWidget(self.n_slider)

        # Force slider
        self.force_label = QLabel("Force [N]: 50")
        self.force_label.setFont(font_label)
        self.force_slider = QSlider(Qt.Horizontal)
        self.force_slider.setMinimum(0)
        self.force_slider.setMaximum(100)
        self.force_slider.setValue(50)
        self.force_slider.setTickInterval(10)
        self.force_slider.setTickPosition(QSlider.TicksBelow)
        self.force_slider.valueChanged.connect(self._update_force_label)
        layout.addWidget(self.force_label)
        layout.addWidget(self.force_slider)

        # Heat slider
        self.heat_label = QLabel("Heat [°C]: 50")
        self.heat_label.setFont(font_label)
        self.heat_slider = QSlider(Qt.Horizontal)
        self.heat_slider.setMinimum(0)
        self.heat_slider.setMaximum(100)
        self.heat_slider.setValue(50)
        self.heat_slider.setTickInterval(10)
        self.heat_slider.setTickPosition(QSlider.TicksBelow)
        self.heat_slider.valueChanged.connect(self._update_heat_label)
        layout.addWidget(self.heat_label)
        layout.addWidget(self.heat_slider)

        # FEM checkbox
        self.fem_checkbox = QCheckBox("Enable FEM Analysis")
        self.fem_checkbox.setFont(font_label)
        layout.addWidget(self.fem_checkbox)

        # Spacer before button
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Run button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(self.run_button)

    # --- Internal label updates ---
    def _update_n_label(self, value):
        self.n_label.setText(f"n-gon: {value} corners")

    def _update_force_label(self, value):
        self.force_label.setText(f"Force [N]: {value}")

    def _update_heat_label(self, value):
        self.heat_label.setText(f"Heat [°C]: {value}")
