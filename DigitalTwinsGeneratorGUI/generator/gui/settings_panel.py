# generator/gui/settings_panel.py

from PySide6.QtWidgets import (
    QWidget,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
)
from PySide6.QtCore import Qt, Signal


class SettingsPanel(QWidget):
    """
    Settings panel for the Telemetry Generator.

    Responsibilities:
        - Configure row count
        - Configure file format (CSV / Parquet)
        - Configure target file size (GB)
        - Configure sampling frequency
        - Emit 'generate_clicked' when user presses Generate
    """

    generate_clicked = Signal()

    def __init__(self):
        super().__init__()
        self._build_ui()

    # ---------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()

        # -----------------------------------------------------
        # Row Count
        # -----------------------------------------------------
        row_group = QGroupBox("Row Count")
        row_layout = QVBoxLayout()

        self.row_spin = QSpinBox()
        self.row_spin.setRange(1_000, 100_000_000)
        self.row_spin.setValue(10_000_000)

        row_layout.addWidget(QLabel("Number of rows:"))
        row_layout.addWidget(self.row_spin)
        row_group.setLayout(row_layout)

        # -----------------------------------------------------
        # File Format
        # -----------------------------------------------------
        format_group = QGroupBox("File Format")
        format_layout = QVBoxLayout()

        self.csv_radio = QRadioButton("CSV")
        self.parquet_radio = QRadioButton("Parquet")
        self.csv_radio.setChecked(True)

        format_layout.addWidget(self.csv_radio)
        format_layout.addWidget(self.parquet_radio)
        format_group.setLayout(format_layout)

        # -----------------------------------------------------
        # Target File Size (GB)
        # -----------------------------------------------------
        size_group = QGroupBox("Target File Size")
        size_layout = QVBoxLayout()

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 50)   # 1–50 GB
        self.size_slider.setValue(10)

        self.size_label = QLabel("10 GB")

        # Connect slider to label update
        self.size_slider.valueChanged.connect(self.update_size_label)

        size_layout.addWidget(QLabel("Desired file size:"))
        size_layout.addWidget(self.size_slider)
        size_layout.addWidget(self.size_label)
        size_group.setLayout(size_layout)

        # -----------------------------------------------------
        # Sampling Frequency
        # -----------------------------------------------------
        freq_group = QGroupBox("Sampling Frequency")
        freq_layout = QVBoxLayout()

        self.freq_spin = QSpinBox()
        self.freq_spin.setRange(1, 10_000)
        self.freq_spin.setValue(10)
        self.freq_spin.setSuffix(" Hz")

        freq_layout.addWidget(QLabel("Frequency (Hz):"))
        freq_layout.addWidget(self.freq_spin)
        freq_group.setLayout(freq_layout)

        # -----------------------------------------------------
        # Generate Button
        # -----------------------------------------------------
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self._on_generate_clicked) 

        # -----------------------------------------------------
        # Assemble Layout
        # -----------------------------------------------------
        layout.addWidget(row_group)
        layout.addWidget(format_group)
        layout.addWidget(size_group)
        layout.addWidget(freq_group)
        layout.addWidget(self.generate_btn)
        layout.addStretch()

        self.setLayout(layout)

    # ---------------------------------------------------------
    # Slider Label Update
    # ---------------------------------------------------------
    def update_size_label(self, value):
        """
        Updates the GB label next to the file size slider.
        """
        self.size_label.setText(f"{value} GB")

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def get_config(self) -> dict:
        """
        Returns a dictionary with all generator settings.
        """
        return {
            "rows": self.row_spin.value(),
            "file_format": "csv" if self.csv_radio.isChecked() else "parquet",
            "target_gb": self.size_slider.value(),
            "frequency_hz": self.freq_spin.value(),
        }
        
    def _on_generate_clicked(self):
        print("GENERATE BUTTON CLICKED")
        self.generate_clicked.emit()