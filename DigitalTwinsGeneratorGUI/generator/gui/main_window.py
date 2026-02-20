# generator/gui/main_window.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter
)
from PySide6.QtCore import Qt

from .schema_panel import SchemaPanel
from .settings_panel import SettingsPanel
from .preview_panel import PreviewPanel
from .status_bar import StatusBar

from ..core.generator import TelemetryGenerator


class MainWindow(QMainWindow):
    """
    Main GUI window for the Telemetry Generator.
    Assembles all panels, manages layout, and orchestrates
    communication between GUI widgets and the generator backend.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Telemetry Data Generator")
        self.setMinimumSize(1200, 700)

        # --- Panels ---
        self.schema_panel = SchemaPanel()
        self.settings_panel = SettingsPanel()
        self.preview_panel = PreviewPanel()
        self.status_bar = StatusBar()

        # --- Generator backend (initialized later) ---
        self.generator = None

        # --- Layout ---
        self._build_layout()

        # --- Connections ---
        self._connect_signals()

    # ---------------------------------------------------------
    # Layout Assembly
    # ---------------------------------------------------------
    def _build_layout(self):
        """
        Creates the main layout using a horizontal splitter:
        [Schema Panel | Settings Panel | Live Preview Panel]
        """
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.schema_panel)
        splitter.addWidget(self.settings_panel)
        splitter.addWidget(self.preview_panel)

        splitter.setSizes([300, 300, 600])  # initial proportions

        main_layout.addWidget(splitter)
        main_layout.addWidget(self.status_bar)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    # ---------------------------------------------------------
    # Signal Connections
    # ---------------------------------------------------------
    def _connect_signals(self):
        """
        Connects GUI events to backend logic.
        """
        print("CONNECTING SIGNALS")
        self.settings_panel.generate_clicked.connect(self._start_generation)

        # NEW: update GB label when slider moves
        self.settings_panel.size_slider.valueChanged.connect(
            self.settings_panel.update_size_label
        )

    # ---------------------------------------------------------
    # Generation Logic
    # ---------------------------------------------------------
    def _start_generation(self):
        """
        Triggered when the user presses the 'Generate' button.
        Collects schema + settings, initializes the generator backend,
        and starts the simulation loop.
        """
        print("START_GENERATION CALLED")
        try:
            schema = self.schema_panel.get_schema()
            config = self.settings_panel.get_config()
            print("SCHEMA:", schema)
            print("CONFIG:", config)

            self.generator = TelemetryGenerator(
                schema=schema,
                config=config,
                preview_callback=self.preview_panel.update_preview,
                progress_callback=self.status_bar.update_progress,
                alert_callback=self.status_bar.show_alert
            )

            self.status_bar.show_message("Starting data generation...")
            self.generator.start()
            print("GENERATOR START CALLED")
        except Exception as e:
            print("ERROR IN _start_generation:", e)
            import traceback
            traceback.print_exc()


    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    def closeEvent(self, event):
        """
        Ensures generator thread stops cleanly when window closes.
        """
        if self.generator:
            self.generator.stop()
        event.accept()