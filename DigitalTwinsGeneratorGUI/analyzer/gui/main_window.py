# analyzer/gui/main_window.py

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QMessageBox,
)
from PySide6.QtCore import Qt

import os

from .settings_panel import SettingsPanel
from .module_panel import ModulePanel
from .visualization_tabs import VisualizationTabs
from .log_panel import LogPanel
from .health_summary import HealthSummary
from .progress_bar import ProgressBar

from ..core.analyzer_loop import AnalyzerLoop
from ..core.config_loader import load_config
from ..core.alert_listener import AlertListener


class MainWindow(QMainWindow):
    """
    Main window for the Telemetry Analyzer GUI.
    """

    def __init__(self, config_path=None):
        super().__init__()
        self.config_path = config_path

        # ---------------------------------------------------------
        # Load configuration FIRST
        # ---------------------------------------------------------
        if self.config_path:
            config_path = os.path.abspath(self.config_path)
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config.json")

        self.config = self._safe_load_config(config_path)

        # ---------------------------------------------------------
        # GUI setup
        # ---------------------------------------------------------
        self.setWindowTitle("Telemetry Data Analyzer")
        self.setMinimumSize(1400, 800)

        self.settings_panel = SettingsPanel(self.config)
        self.module_panel = ModulePanel()
        self.visualization_tabs = VisualizationTabs()
        self.log_panel = LogPanel()
        self.health_summary = HealthSummary()
        self.progress_bar = ProgressBar()

        self.analyzer: AnalyzerLoop | None = None
        self.alert_listener: AlertListener | None = None

        self._build_layout()
        self._connect_signals()
        self._init_backend()

    # ---------------------------------------------------------
    # Config loading
    # ---------------------------------------------------------
    def _safe_load_config(self, path: str):
        try:
            return load_config(path)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Config Error",
                f"Could not load {path}.\n\n{e}\n\n"
                "You can still start the Analyzer, but some features may be disabled.",
            )
            return {}

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------
    def _build_layout(self):
        central = QWidget()
        main_layout = QVBoxLayout()

        splitter = QSplitter(Qt.Horizontal)

        # Left side
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.settings_panel)
        left_layout.addWidget(self.module_panel)
        left_layout.addStretch()
        left_widget.setLayout(left_layout)

        # Right side
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.visualization_tabs)
        right_layout.addWidget(self.health_summary)
        right_layout.addWidget(self.log_panel)
        right_widget.setLayout(right_layout)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        # Bottom progress bar
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.progress_bar)

        main_layout.addWidget(splitter)
        main_layout.addLayout(bottom_layout)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    # ---------------------------------------------------------
    # Signal wiring
    # ---------------------------------------------------------
    def _connect_signals(self):
        self.settings_panel.start_requested.connect(self._start_analysis)
        self.settings_panel.stop_requested.connect(self._stop_analysis)
        self.module_panel.modules_changed.connect(self._update_modules)

    # ---------------------------------------------------------
    # Backend initialization
    # ---------------------------------------------------------
    def _init_backend(self):
        self.analyzer = AnalyzerLoop(
            config=self.config,
            progress_callback=self.progress_bar.update_progress,
            log_callback=self.log_panel.append_log,
            visualization_callback=self.visualization_tabs.update_visualizations,
            health_callback=self.health_summary.update_health,
        )

        self.alert_listener = AlertListener(
            host=self.config.get("alerts", {}).get("socket_host", "127.0.0.1"),
            port=self.config.get("alerts", {}).get("socket_port", 5050),
            enabled=self.config.get("alerts", {}).get("socket_enabled", True),
            alert_callback=self._handle_alert,
        )
        self.alert_listener.start()

    # ---------------------------------------------------------
    # Analysis control
    # ---------------------------------------------------------
    def _start_analysis(self):
        if not self.analyzer:
            return

        analysis_config = self.settings_panel.get_analysis_config()
        selected_modules = self.module_panel.get_selected_modules()

        self.log_panel.append_log("Starting analysis...")
        self.analyzer.start(analysis_config, selected_modules)

    def _stop_analysis(self):
        if not self.analyzer:
            return

        self.log_panel.append_log("Stopping analysis...")
        self.analyzer.stop()

    def _update_modules(self, modules):
        if not self.analyzer:
            return
        self.analyzer.set_modules(modules)

    # ---------------------------------------------------------
    # Alert handling
    # ---------------------------------------------------------
    def _handle_alert(self, alert: dict):
        event = alert.get("event", "unknown")
        payload = alert.get("payload", {})

        self.log_panel.append_log(f"[ALERT] {event} - {payload}")

        if event == "chunk_written":
            self.health_summary.mark_data_updated()
        elif event == "generation_complete":
            self.health_summary.mark_generation_complete()

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    def closeEvent(self, event):
        if self.analyzer:
            self.analyzer.stop()
        if self.alert_listener:
            self.alert_listener.stop()
        super().closeEvent(event)