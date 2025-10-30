import sys
import pandas as pd
import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QListWidget, QListWidgetItem,
    QFileDialog, QCheckBox, QFrame, QStatusBar, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal

from plugin_loader import discover_plugins
from schema_io import validate_schema, parse_schema, load_yaml_schema, save_yaml_schema
from synthetic_models import generate_synthetic_data
from risk_models import risk_pipeline
from physics_models import physics_pipeline


class QuantCanvas(QMainWindow):
    theme_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FIN QuantLab — QuantCanvas")
        self.resize(1400, 900)
        self.df = None
        self.raw_df = None
        self.schema = {}
        self.simulation_schema = {}

        discover_plugins()

        central = QWidget()
        self.layout = QVBoxLayout()
        central.setLayout(self.layout)
        self.setCentralWidget(central)

        controls = QHBoxLayout()
        load_btn = QPushButton("Load CSV")
        load_btn.clicked.connect(self.load_csv)
        import_yaml_btn = QPushButton("Import YAML")
        import_yaml_btn.clicked.connect(self.import_yaml_schema)
        export_yaml_btn = QPushButton("Export YAML")
        export_yaml_btn.clicked.connect(self.export_yaml_schema)
        self.theme_toggle = QCheckBox("Dark Mode")
        self.theme_toggle.stateChanged.connect(self.toggle_theme)
        controls.addWidget(load_btn)
        controls.addWidget(import_yaml_btn)
        controls.addWidget(export_yaml_btn)
        controls.addWidget(self.theme_toggle)
        self.layout.addLayout(controls)

        self.asset_list = QListWidget()
        self.asset_list.setDragEnabled(True)
        self.asset_list.setSelectionMode(QListWidget.MultiSelection)
        self.asset_list.setFixedHeight(100)
        self.layout.addWidget(QLabel("Assets"))
        self.layout.addWidget(self.asset_list)
        self.asset_list.itemSelectionChanged.connect(self.update_chart)

        self.widget_panel = QFrame()
        self.widget_layout = QVBoxLayout()
        self.widget_panel.setLayout(self.widget_layout)
        self.layout.addWidget(QLabel("Schema Widgets"))
        self.layout.addWidget(self.widget_panel)

        self.chart = pg.PlotWidget()
        self.chart.setTitle("Real-Time Chart")
        self.chart.setLabel("left", "Price")
        self.chart.setLabel("bottom", "Time", units="s")
        self.chart.addLegend(offset=(80, 20), anchor=(0, 0))
        self.layout.addWidget(self.chart)

        status_bar = QStatusBar()
        status_bar.addWidget(QLabel("Ready"))
        self.progress = QProgressBar()
        self.progress.setValue(0)
        status_bar.addPermanentWidget(self.progress)
        self.setStatusBar(status_bar)

    def toggle_theme(self, state):
        if state == Qt.Checked:
            self.setStyleSheet("background-color: #2b2b2b; color: white;")
        else:
            self.setStyleSheet("")

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            self.progress.setValue(10)
            self.df = pd.read_csv(path, parse_dates=True)
            self.raw_df = self.df.copy()
            try:
                validate_schema(self.df)
            except Exception as e:
                print(f"Schema validation failed: {e}")
                return
            self.schema = parse_schema(self.df)
            self.populate_assets()
            self.generate_widgets()
            self.update_chart()
            self.progress.setValue(100)

    def import_yaml_schema(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import YAML", "", "YAML Files (*.yaml *.yml)")
        if not path:
            return

        self.progress.setValue(5)
        schema = load_yaml_schema(path)

        sim_config = schema.get("simulation", {})
        self.simulation_schema = sim_config
        try:
            self.df = generate_synthetic_data(**sim_config)
            self.raw_df = self.df.copy()
            self.autosave_csv(self.df, label="simulation")
        except Exception as e:
            print(f"Simulation failed: {e}")
            return

        self.progress.setValue(25)
        try:
            validate_schema(self.df)
        except Exception as e:
            print(f"Schema validation failed: {e}")
            return

        self.schema = parse_schema(self.df)
        self.populate_assets()
        self.generate_widgets()
        self.update_chart()
        self.progress.setValue(60)

        analysis_config = schema.get("analysis", {})
        model_type = analysis_config.get("type", "risk")
        model_name = analysis_config.get("model")

        if model_name:
            try:
                if model_type == "risk":
                    returns = np.log(self.df / self.df.shift(1)).dropna()
                    result = risk_pipeline(returns, model_name)
                elif model_type == "physics":
                    result = physics_pipeline(self.df, model_name)
                else:
                    print(f"Unknown analysis type: {model_type}")
                    return

                print(f"{model_type.capitalize()} analysis result for {model_name}:\n", result.head())
                if isinstance(result, pd.DataFrame):
                    self.df = result
                    self.schema = parse_schema(self.df)
                    self.populate_assets()
                    self.update_chart()
                    self.autosave_csv(self.df, label="analysis")
                self.progress.setValue(100)
            except Exception as e:
                print(f"Analysis failed: {e}")

    def export_yaml_schema(self):
        if not self.simulation_schema:
            print("No schema to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export YAML", "", "YAML Files (*.yaml *.yml)")
        if path:
            save_yaml_schema(self.simulation_schema, path)
            print(f"Exported YAML schema to {path}")

    def populate_assets(self):
        self.asset_list.clear()
        self.asset_list.blockSignals(True)  # prevent premature triggers
        for col in self.df.columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsDragEnabled)
            self.asset_list.addItem(item)
        self.asset_list.blockSignals(False)

        # ✅ Explicitly select all items after population
        for i in range(self.asset_list.count()):
            self.asset_list.item(i).setSelected(True)

    def generate_widgets(self):
        # Clear existing widgets
        for i in reversed(range(self.widget_layout.count())):
            widget = self.widget_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.metric_labels = {}  # Store value labels per asset

        for col, dtype in self.schema.items():
            label = QLabel(f"{col} ({dtype})")
            label.setStyleSheet("font-weight: bold;")
            self.widget_layout.addWidget(label)

            if dtype == "numeric":
                combo = QComboBox()
                combo.addItems(["mean", "std", "min", "max"])
                combo.currentIndexChanged.connect(lambda _, c=col, w=combo: self.apply_metric(c, w))
                self.widget_layout.addWidget(combo)

                value_label = QLabel("Value: —")
                value_label.setStyleSheet("color: gray; margin-left: 10px;")
                self.widget_layout.addWidget(value_label)
                self.metric_labels[col] = value_label

            elif dtype == "categorical":
                combo = QComboBox()
                combo.addItems(self.df[col].astype(str).unique().tolist())
                combo.currentIndexChanged.connect(lambda _, c=col, w=combo: self.filter_category(c, w))
                self.widget_layout.addWidget(combo)

            elif dtype == "datetime":
                self.widget_layout.addWidget(QLabel("Time filter coming soon..."))

    def apply_metric(self, column, widget):
        metric = widget.currentText()
        if metric and column in self.df.columns:
            try:
                value = getattr(self.df[column], metric)()
                formatted = f"{value:.4f}" if isinstance(value, (int, float, np.number)) else str(value)
                print(f"{metric} of {column}: {formatted}")
                if column in self.metric_labels:
                    self.metric_labels[column].setText(f"Value: {formatted}")
            except Exception as e:
                print(f"Failed to compute {metric} for {column}: {e}")
                if column in self.metric_labels:
                    self.metric_labels[column].setText("Value: error")

    def filter_category(self, column, widget):
        value = widget.currentText()
        if value:
            filtered = self.df[self.df[column].astype(str) == value]
            print(f"Filtered {column} = {value}, {len(filtered)} rows")

    def autosave_csv(self, df, label="output"):
        import os
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}.csv"
        save_dir = "autosaves"
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)

        try:
            df.to_csv(path, index=False)
            print(f"Auto-saved {label} to {path}")
        except Exception as e:
            print(f"Failed to save {label}: {e}")

    def update_chart(self):
        if self.df is None or self.df.empty:
            print("No data to plot.")
            return

        selected_items = self.asset_list.selectedItems()
        selected_cols = [item.text() for item in selected_items if item.text() in self.df.columns]

        print("Selected columns:", selected_cols)
        print("Available columns:", self.df.columns.tolist())

        if not selected_cols:
            print("No assets selected.")
            return

        self.chart.clear()

        # Re-add legend with offset to avoid overlap
        self.chart.addLegend(offset=(80, 20), anchor=(0, 0))

        # Define color palette
        COLOR_PALETTE = [
            (255, 0, 0),      # Red
            (0, 128, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 165, 0),    # Orange
            (128, 0, 128),    # Purple
            (0, 206, 209),    # Turquoise
            (255, 192, 203),  # Pink
            (128, 128, 128),  # Gray
            (255, 215, 0),    # Gold
            (0, 0, 0)         # Black
        ]

        for i, col in enumerate(selected_cols):
            series = self.df[col].dropna()
            if not np.issubdtype(series.dtype, np.number):
                print(f"Skipping non-numeric column: {col}")
                continue
            if series.empty:
                print(f"No valid data to plot for {col}")
                continue
            x = np.arange(len(series))
            y = series.values
            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            pen = pg.mkPen(color=color, width=2)
            print(f"Plotting {col}: {len(y)} points with color {color}")
            self.chart.plot(x, y, pen=pen, name=col)
