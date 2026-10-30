# mlflow_local_runner/src/gui/app_window.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QMessageBox
)
from PySide6.QtCore import Signal, Slot

from gui.upload_panel import UploadPanel
from gui.config_panel import ConfigPanel
from gui.run_panel import RunPanel
from gui.results_panel import ResultsPanel

from core.runner import Runner
from utils.logger import get_logger

import subprocess
import threading
import time
import psutil
import sys
import os


class AppWindow(QMainWindow):

    run_finished = Signal(dict)
    run_failed = Signal(str)

    def __init__(self, config: dict):
        super().__init__()

        self.logger = get_logger(__name__)
        self.logger.info("Initialisiere AppWindow...")

        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

        self.setWindowTitle("MLflow Local Runner")
        self.resize(1200, 800)

        self.config = config or {}
        self.script_path = None
        self.dataset_path = None

        self.runner = Runner()

        self._init_ui()
        self._connect_signals()
        self._connect_runner_logging()

        threading.Thread(target=self._ensure_mlflow_running, daemon=True).start()

    # ---------------------------------------------------------
    # RUNNER-LOGSIGNALS MIT GUI VERBINDEN
    # ---------------------------------------------------------

    def _connect_runner_logging(self):
        self.runner.on_log_info = self.run_panel.append_info
        self.runner.on_log_warning = self.run_panel.append_warning
        self.runner.on_log_error = self.run_panel.append_error

    # ---------------------------------------------------------
    # MLflow START
    # ---------------------------------------------------------

    def _free_port_5000(self):
        self.logger.info("Prüfe, ob Port 5000 blockiert ist...")
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                for conn in proc.connections(kind="inet"):
                    if conn.laddr.port == 5000 and proc.pid != 0:
                        self.logger.warning(
                            f"Beende Prozess auf Port 5000: PID={proc.pid}, Name={proc.info['name']}"
                        )
                        proc.kill()
                        time.sleep(0.3)
            except Exception:
                pass

    def _start_mlflow_server(self):
        self.logger.info("Starte MLflow-Server...")

        cmd = [
            sys.executable, "-m", "mlflow",
            "server",
            "--host", "0.0.0.0",
            "--port", "5000",
            "--workers", "1",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "./mlruns",
        ]

        env = os.environ.copy()
        env["MLFLOW_ENABLE_SECURITY_MIDDLEWARE"] = "false"

        try:
            subprocess.Popen(cmd, env=env)
        except Exception as e:
            self.logger.error(f"MLflow-Server konnte nicht gestartet werden: {e}")

    def _ensure_mlflow_running(self):
        self._free_port_5000()
        self._start_mlflow_server()
        self.logger.info("MLflow wurde gestartet (Healthcheck deaktiviert).")

    # ---------------------------------------------------------
    # GUI INITIALISIERUNG
    # ---------------------------------------------------------

    def _init_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.upload_panel = UploadPanel()
        self.config_panel = ConfigPanel(self.config)
        self.run_panel = RunPanel()
        self.results_panel = ResultsPanel()

        self.tabs.addTab(self.upload_panel, "Upload")
        self.tabs.addTab(self.config_panel, "Konfiguration")
        self.tabs.addTab(self.run_panel, "Run")
        self.tabs.addTab(self.results_panel, "Ergebnisse")

    # ---------------------------------------------------------
    # SIGNALVERBINDUNGEN
    # ---------------------------------------------------------

    def _connect_signals(self):
        self.upload_panel.script_selected.connect(self._on_script_selected)
        self.upload_panel.dataset_selected.connect(self._on_dataset_selected)
        self.run_panel.start_run_clicked.connect(self._on_start_run)
        self.run_finished.connect(self._on_run_finished)
        self.run_failed.connect(self._on_run_failed)

    # ---------------------------------------------------------
    # EVENT-HANDLER
    # ---------------------------------------------------------

    @Slot(str)
    def _on_script_selected(self, path: str):
        self.script_path = path
        self.logger.info(f"Skript ausgewählt: {path}")

    @Slot(str)
    def _on_dataset_selected(self, path: str):
        self.dataset_path = path
        self.logger.info(f"Dataset ausgewählt: {path}")

    @Slot()
    def _on_start_run(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "Fehlender Datensatz",
                                "Bitte wählen Sie ein Dataset (.csv) aus.")
            return

        # -----------------------------------------------------
        # NEU: Projektname, Experimentname, Runname übernehmen
        # -----------------------------------------------------
        mlflow_config = self.config_panel.get_config()

        self.logger.info("Starte Run...")
        self.logger.info(f"MLflow-Konfiguration: {mlflow_config}")

        # Run-Tab aktivieren
        self.tabs.setCurrentWidget(self.run_panel)
        self.run_panel.clear_log()
        self.run_panel.set_running(True)

        def _worker():
            try:
                results = self.runner.run(
                    script_path=self.script_path,
                    dataset_path=self.dataset_path,
                    mlflow_config=mlflow_config
                )
                self.run_finished.emit(results)
            except Exception as e:
                self.run_failed.emit(str(e))

        threading.Thread(target=_worker, daemon=True).start()

    @Slot(dict)
    def _on_run_finished(self, metrics: dict):
        self.run_panel.set_running(False)
        self.results_panel.display_results(metrics)
        self.tabs.setCurrentWidget(self.results_panel)

    @Slot(str)
    def _on_run_failed(self, error_message: str):
        self.run_panel.set_running(False)
        QMessageBox.critical(self, "Run fehlgeschlagen", error_message)
        self.logger.error(error_message)
