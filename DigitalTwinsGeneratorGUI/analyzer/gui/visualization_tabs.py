# analyzer/gui/visualization_tabs.py

from PySide6.QtWidgets import (
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QLabel,
)
from PySide6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class VisualizationTabs(QWidget):
    """
    Tabbed visualization interface for the Telemetry Analyzer.

    Responsibilities:
        - Provide separate tabs for each analysis module
        - Expose update_*() methods for AnalyzerLoop to push results
        - Render plots using Matplotlib
        - Keep GUI decoupled from analysis logic

    Tabs:
        - Time Series
        - Clustering
        - Forecasting
        - NLP
        - Deep Learning
        - XAI
    """

    def __init__(self):
        super().__init__()
        self._build_ui()

    # ---------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()

        self.tabs = QTabWidget()

        # --- Time Series Tab ---
        self.ts_fig = Figure(figsize=(5, 3))
        self.ts_canvas = FigureCanvas(self.ts_fig)
        self.ts_ax = self.ts_fig.add_subplot(111)
        ts_widget = QWidget()
        ts_layout = QVBoxLayout()
        ts_layout.addWidget(self.ts_canvas)
        ts_widget.setLayout(ts_layout)
        self.tabs.addTab(ts_widget, "Time Series")

        # --- Clustering Tab ---
        self.cluster_fig = Figure(figsize=(5, 3))
        self.cluster_canvas = FigureCanvas(self.cluster_fig)
        self.cluster_ax = self.cluster_fig.add_subplot(111)
        cluster_widget = QWidget()
        cluster_layout = QVBoxLayout()
        cluster_layout.addWidget(self.cluster_canvas)
        cluster_widget.setLayout(cluster_layout)
        self.tabs.addTab(cluster_widget, "Clustering")

        # --- Forecasting Tab ---
        self.forecast_fig = Figure(figsize=(5, 3))
        self.forecast_canvas = FigureCanvas(self.forecast_fig)
        self.forecast_ax = self.forecast_fig.add_subplot(111)
        forecast_widget = QWidget()
        forecast_layout = QVBoxLayout()
        forecast_layout.addWidget(self.forecast_canvas)
        forecast_widget.setLayout(forecast_layout)
        self.tabs.addTab(forecast_widget, "Forecasting")

        # --- NLP Tab ---
        self.nlp_label = QLabel("NLP results will appear here.")
        self.nlp_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        nlp_widget = QWidget()
        nlp_layout = QVBoxLayout()
        nlp_layout.addWidget(self.nlp_label)
        nlp_widget.setLayout(nlp_layout)
        self.tabs.addTab(nlp_widget, "NLP")

        # --- Deep Learning Tab ---
        self.dl_fig = Figure(figsize=(5, 3))
        self.dl_canvas = FigureCanvas(self.dl_fig)
        self.dl_ax = self.dl_fig.add_subplot(111)
        dl_widget = QWidget()
        dl_layout = QVBoxLayout()
        dl_layout.addWidget(self.dl_canvas)
        dl_widget.setLayout(dl_layout)
        self.tabs.addTab(dl_widget, "Deep Learning")

        # --- XAI Tab ---
        self.xai_label = QLabel("XAI feature attributions will appear here.")
        self.xai_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        xai_widget = QWidget()
        xai_layout = QVBoxLayout()
        xai_layout.addWidget(self.xai_label)
        xai_widget.setLayout(xai_layout)
        self.tabs.addTab(xai_widget, "XAI")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    # ---------------------------------------------------------
    # Update Methods (called by AnalyzerLoop)
    # ---------------------------------------------------------
    def update_visualizations(self, results: dict):
        """
        Central update entry point.
        AnalyzerLoop sends a dict with keys:
            - "time_series"
            - "clustering"
            - "forecasting"
            - "nlp"
            - "deep_learning"
            - "xai"
        Each key maps to module-specific results.
        """
        if "time_series" in results:
            self.update_time_series(results["time_series"])

        if "clustering" in results:
            self.update_clustering(results["clustering"])

        if "forecasting" in results:
            self.update_forecasting(results["forecasting"])

        if "nlp" in results:
            self.update_nlp(results["nlp"])

        if "deep_learning" in results:
            self.update_deep_learning(results["deep_learning"])

        if "xai" in results:
            self.update_xai(results["xai"])

    # ---------------------------------------------------------
    # Individual Update Methods
    # ---------------------------------------------------------
    def update_time_series(self, data):
        """
        Expects:
            data = {
                "x": [...],
                "y": [...],
                "label": "Temperature"
            }
        """
        self.ts_ax.clear()
        self.ts_ax.plot(data["x"], data["y"], label=data.get("label", ""))
        self.ts_ax.set_title("Time Series")
        self.ts_ax.legend()
        self.ts_canvas.draw_idle()

    def update_clustering(self, data):
        """
        Expects:
            data = {
                "x": [...],
                "y": [...],
                "labels": [...]
            }
        """
        self.cluster_ax.clear()
        self.cluster_ax.scatter(data["x"], data["y"], c=data["labels"], cmap="viridis")
        self.cluster_ax.set_title("Clustering")
        self.cluster_canvas.draw_idle()

    def update_forecasting(self, data):
        """
        Expects:
            data = {
                "history_x": [...],
                "history_y": [...],
                "forecast_x": [...],
                "forecast_y": [...]
            }
        """
        self.forecast_ax.clear()
        self.forecast_ax.plot(data["history_x"], data["history_y"], label="History")
        self.forecast_ax.plot(data["forecast_x"], data["forecast_y"], label="Forecast")
        self.forecast_ax.set_title("Forecasting")
        self.forecast_ax.legend()
        self.forecast_canvas.draw_idle()

    def update_nlp(self, text: str):
        """
        Expects:
            text = "Summary of log messages..."
        """
        self.nlp_label.setText(text)

    def update_deep_learning(self, data):
        """
        Expects:
            data = {
                "x": [...],
                "anomaly_score": [...]
            }
        """
        self.dl_ax.clear()
        self.dl_ax.plot(data["x"], data["anomaly_score"], color="red")
        self.dl_ax.set_title("Anomaly Score")
        self.dl_canvas.draw_idle()

    def update_xai(self, text: str):
        """
        Expects:
            text = "Feature attribution summary..."
        """
        self.xai_label.setText(text)