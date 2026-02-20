# generator/gui/preview_panel.py

import numpy as np
from collections import deque

from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLabel, QComboBox
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PreviewPanel(QWidget):
    MAX_POINTS = 500

    def __init__(self):
        super().__init__()

        self.setMinimumWidth(400)

        self.buffers = {}
        self.current_column = None

        self._build_ui()
        self._setup_plot()

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()

        group = QGroupBox("Live Preview")
        group_layout = QVBoxLayout()

        self.column_selector = QComboBox()
        self.column_selector.currentTextChanged.connect(self._change_column)

        group_layout.addWidget(QLabel("Preview Column:"))
        group_layout.addWidget(self.column_selector)

        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)

        group_layout.addWidget(self.canvas)
        group.setLayout(group_layout)

        layout.addWidget(group)
        self.setLayout(layout)

    # ---------------------------------------------------------
    # Plot Setup
    # ---------------------------------------------------------
    def _setup_plot(self):
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Live Data Stream")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Value")
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.grid(True)
        # Ensure labels and titles are fully visible
        self.figure.tight_layout()

    # ---------------------------------------------------------
    # Column Selection
    # ---------------------------------------------------------
    def set_available_columns(self, columns, sample_row=None):
        """
        Populate dropdown with numeric-only columns.
        sample_row: a dict from the preview callback (one row of data)
        """
        self.column_selector.clear()

        numeric_cols = []

        for col in columns:
            if sample_row is None:
                # Fallback: assume numeric until proven otherwise
                numeric_cols.append(col)
            else:
                val = sample_row.get(col)
                if isinstance(val, (int, float, np.number)):
                    numeric_cols.append(col)

        for col in numeric_cols:
            self.column_selector.addItem(col)

        if numeric_cols:
            self.current_column = numeric_cols[0]

    def _change_column(self, col_name):
        self.current_column = col_name

        if col_name not in self.buffers:
            self.buffers[col_name] = deque(maxlen=self.MAX_POINTS)

        # Reset the line only — do NOT clear axes
        self.line.set_data([], [])

        self.ax.set_title(f"Live Data Stream: {col_name}")
        self.ax.set_ylabel(col_name)

        self.canvas.draw()

    # ---------------------------------------------------------
    # Preview Update
    # ---------------------------------------------------------
    def update_preview(self, data_dict):
        if not data_dict:
            return

        # Initialize dropdown on first update
        if self.column_selector.count() == 0:
           self.set_available_columns(list(data_dict.keys()), sample_row=data_dict)

        if not self.current_column:
            return

        if self.current_column not in data_dict:
            return

        value = data_dict[self.current_column]

        # Initialize buffer if needed
        if self.current_column not in self.buffers:
            self.buffers[self.current_column] = deque(maxlen=self.MAX_POINTS)

        self.buffers[self.current_column].append(value)

        y = list(self.buffers[self.current_column])
        x = list(range(len(y)))

        # Update the existing line — do NOT clear axes
        self.line.set_data(x, y)

        # Adjust axes limits
        self.ax.set_xlim(max(0, len(y) - self.MAX_POINTS), max(self.MAX_POINTS, len(y)))

        y_min = min(y)
        y_max = max(y)
        if y_max == y_min:
            y_max = y_min + 1.0
        self.ax.set_ylim(y_min - 1, y_max + 1)

        self.canvas.draw()