# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  gui/main_window.py
ROLE:    Main supervisory dashboard orchestrating asynchronous execution threads,
         telemetric view updates, session lookups, and unified Markdown exports.
================================================================================
"""

import os
import json
import logging
import uuid
from datetime import datetime

from PyQt6.QtWidgets import (QMainWindow, QWidget, QTextEdit, QTextBrowser,
                             QPushButton, QLabel, QHBoxLayout, QVBoxLayout,
                             QFileDialog, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QFile

# Import out of our custom architectural layers
from core.database import SessionPersistenceManager
from core.router import IntentPreRoutingEngine
from core.metrics import QuantumRGDriftAnalyzer, HeterogeneousDiagnosticEngine
from api.web_bridge import BoundedWebSessionBridge
from api.thread_manager import ThreadContextOrchestrator

# Import peer View widgets and sub-dialog window structures
from gui.widgets import QuantumHallucinationShieldWidget, MetricsTelemetryGridPanel
from gui.history_window import ChatHistoryDialog
from gui.auth_window import AccountStatusCenterDialog

log = logging.getLogger("MainWindowUI")


class AgentExecutionWorker(QThread):
    """
    Decouples network transmissions and non-linear space measurements
    from the main UI thread to safeguard interface responsiveness.
    """
    execution_complete = pyqtSignal(dict)
    execution_failed = pyqtSignal(str)
    progress_update = pyqtSignal(str)

    def __init__(self, orchestrator: ThreadContextOrchestrator, session_id: int, prompt: str, route_meta: dict):
        super().__init__()
        self.orchestrator = orchestrator
        self.session_id = session_id
        self.prompt = prompt
        self.route_meta = route_meta

    def run(self):
        """Asynchronous execution channel executing inside background worker threads."""
        try:
            # process_interaction_step now runs N independent generations per
            # prompt (multi-sample self-consistency), so it takes noticeably
            # longer than a single call; progress_callback reports back after
            # each sample so the GUI isn't silent for the whole duration.
            result = self.orchestrator.process_interaction_step(
                self.session_id, self.prompt, self.route_meta,
                progress_callback=lambda msg: self.progress_update.emit(msg)
            )
            self.execution_complete.emit(result)
        except Exception as e:
            log.error(f"Asynchronous worker collapsed during transport phase: {str(e)}")
            self.execution_failed.emit(str(e))


class MainDashboardWindow(QMainWindow):
    """
    Core telemetric workspace aggregating functional controller logic blocks
    and visualizing real-time trajectory analytics inside a custom UI framework.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI ROUTER & HALLUCINATION SHIELD (PyQt6 Web Dashboard)")
        self.resize(1280, 800)

        # 1. Initialize persistent backends and evaluation matrices
        self.db = SessionPersistenceManager()
        self.router = IntentPreRoutingEngine(self.db)
        self.analyzer = QuantumRGDriftAnalyzer()
        self.diagnostic = HeterogeneousDiagnosticEngine(self.analyzer)
        self.bridge = BoundedWebSessionBridge()
        self.orchestrator = ThreadContextOrchestrator(self.db, self.bridge, self.diagnostic)

        # No startup "default_stream" session anymore: now that each prompt
        # is evaluated via self-consistency (a single, self-contained
        # interaction, not a multi-turn thread), every submitted prompt gets
        # its OWN session in _handle_routing_matrix_dispatch below. This is
        # what makes "Verlauf/Chats einsehen" show one row per prompt instead
        # of everything being silently logged under one eternal session, and
        # what makes the metrics plot show only that one interaction's data.
        self.active_session_id = None
        self.active_route_meta = None
        self.last_prompt = ""
        self.last_response = ""
        self.last_diagnostic = None
        self.worker = None

        # 2. Assemble view geometry and link stylesheets
        self._build_dashboard_layout()
        self._load_global_qss_styles()

    # ------------------------------------------------------------------ layout
    def _build_dashboard_layout(self):
        """Assembles structural QWidget trees and maps internal signal networks."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # --- ROW 1: TEXT PROMPT EDITOR ---
        self.prompt_editor = QTextEdit()
        self.prompt_editor.setPlaceholderText("Geben Sie hier Ihre komplexe logische Fragestellung oder Code-Anforderungen ein...")
        self.prompt_editor.setMaximumHeight(110)
        main_layout.addWidget(self.prompt_editor)

        # --- ROW 2: TELEMETRY CONTROL TOOLBAR ---
        control_layout = QHBoxLayout()
        self.btn_execute = QPushButton("Route & Execute")
        self.btn_auth = QPushButton("Account-Status 🔑")

        self.label_routing = QLabel("Zugeordneter Web-Agent: Keine Anfrage aktiv")
        self.label_routing.setStyleSheet("color: #8b949e; font-style: italic; padding-left: 10px;")
        self.label_status = QLabel("System Status: Bereit")
        self.label_status.setStyleSheet("color: #8b949e; font-weight: bold;")

        control_layout.addWidget(self.btn_execute)
        control_layout.addWidget(self.btn_auth)
        control_layout.addWidget(self.label_routing)
        control_layout.addStretch()
        control_layout.addWidget(self.label_status)
        main_layout.addLayout(control_layout)

        # --- ROW 3: DUAL SPOKE GRAPHICAL RESULTS DASHBOARD ---
        dashboard_layout = QHBoxLayout()
        dashboard_layout.setSpacing(15)

        # Column A: Content Output Panel (Markdown Viewer Terminal)
        self.response_viewer = QTextBrowser()
        self.response_viewer.setPlaceholderText("Der verifizierte Antwort-Textstrom aus der echten Web-Sitzung wird hier ausgegeben...")
        dashboard_layout.addWidget(self.response_viewer, stretch=3)

        # Column B: Mathematical Shield Tracking Stack
        shield_column = QVBoxLayout()
        shield_column.setSpacing(10)

        self.shield_widget = QuantumHallucinationShieldWidget()
        self.telemetry_panel = MetricsTelemetryGridPanel()

        shield_column.addWidget(self.shield_widget)
        shield_column.addWidget(self.telemetry_panel)
        dashboard_layout.addLayout(shield_column, stretch=1)

        main_layout.addLayout(dashboard_layout)

        # --- ROW 4: ACTION ACTIONS CORE ---
        footer_layout = QHBoxLayout()
        self.btn_export = QPushButton("Export as GitHub Markdown (.md)")
        self.btn_history = QPushButton("Verlauf/Chats einsehen")
        self.btn_export.setEnabled(False)  # Disabled until text is generated

        footer_layout.addWidget(self.btn_export)
        footer_layout.addStretch()
        footer_layout.addWidget(self.btn_history)
        main_layout.addLayout(footer_layout)

        # Map interaction points to reactive execution slots
        self.btn_execute.clicked.connect(self._handle_routing_matrix_dispatch)
        self.btn_auth.clicked.connect(self._launch_authentication_center)
        self.btn_history.clicked.connect(self._launch_history_dialog)
        self.btn_export.clicked.connect(self._execute_github_markdown_export)

    def _load_global_qss_styles(self):
        """Loads and applies the verified high-contrast external stylesheet."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        qss_path = os.path.join(base_dir, 'styles.qss')
        if os.path.exists(qss_path):
            qss_file = QFile(qss_path)
            if qss_file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
                stream = qss_file.readAll()
                style_sheet = str(stream, encoding='utf-8')
                self.setStyleSheet(style_sheet)
                qss_file.close()
                log.info("Production styles.qss linked and injected smoothly.")

    # ------------------------------------------------------------ execution flow
    def _handle_routing_matrix_dispatch(self):
        """Extracts text inputs, identifies targets via pre-routing, and runs workers."""
        prompt = self.prompt_editor.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Eingabe leer", "Bitte geben Sie einen Prompt ein, bevor Sie die Route berechnen.")
            return

        self.last_prompt = prompt
        self.label_status.setText("Status: Berechne Route...")

        # Compute real-time capability match constraints via the router
        self.active_route_meta = self.router.determine_optimal_route(prompt)
        self.label_routing.setText(f"Zugeordneter Web-Agent: {self.active_route_meta['agent_name']}")
        self.label_status.setText("Status: Erwarte Web-Stream...")

        # A new, distinct prompt gets its own session (one entry in
        # "Verlauf/Chats einsehen"). Resubmitting the EXACT SAME prompt text
        # instead continues the existing session for that prompt, so repeated
        # submissions accumulate as steps and build up a real trajectory in
        # the metrics plot, rather than each one starting a fresh, isolated
        # session with no history of its own.
        existing_id = self.db.find_session_id_for_prompt(prompt)
        if existing_id is not None:
            self.active_session_id = existing_id
        else:
            url_id = f"prompt-{uuid.uuid4().hex[:10]}"
            title = prompt[:80] + ("..." if len(prompt) > 80 else "")
            self.active_session_id = self.db.register_new_web_session(
                url_id, self.active_route_meta["agent_name"], title, self.active_route_meta["chat_type"]
            )

        # Lock core commands to prevent thread overlapping collisions
        self.btn_execute.setEnabled(False)
        self.btn_export.setEnabled(False)

        # Instantiate background computing resource worker loop
        self.worker = AgentExecutionWorker(
            self.orchestrator, self.active_session_id, prompt, self.active_route_meta
        )
        self.worker.execution_complete.connect(self._process_successful_generation)
        self.worker.execution_failed.connect(self._process_failed_generation)
        self.worker.progress_update.connect(lambda msg: self.label_status.setText(f"Status: {msg}"))
        self.worker.start()

    def _process_successful_generation(self, payload: dict):
        """Updates text screens, telemetry grids, and color gauges upon worker return."""
        self.btn_execute.setEnabled(True)
        self.label_status.setText("Status: Bereit")

        self.last_response = payload["response_text"]
        self.last_diagnostic = payload["diagnostic"]
        self.btn_export.setEnabled(True)

        # Render clean text stream straight to the viewer terminal
        self.response_viewer.setText(self.last_response)

        # Extract evaluation slice matrix coordinates
        h_score = self.last_diagnostic["h_score"]
        is_def = self.last_diagnostic["is_definitive_hallucination"]
        drift_x = self.last_diagnostic["mean_drift_x"]
        blur_y = self.last_diagnostic["trace_covariance_y"]
        raw = self.last_diagnostic["raw_metrics"]

        # Push values into the peer widgets (real method names from gui/widgets.py)
        self.shield_widget.update_shield_state(h_score, is_def)
        self.telemetry_panel.render_metrics_package(raw, drift_x, blur_y)

    def _process_failed_generation(self, error_message: str):
        """Restores the UI after a worker failure and informs the user."""
        self.btn_execute.setEnabled(True)
        self.btn_export.setEnabled(bool(self.last_response))
        self.label_status.setText("Status: Fehler")
        log.error(f"Generation failed: {error_message}")
        QMessageBox.critical(
            self,
            "Ausführung fehlgeschlagen",
            f"Die Anfrage konnte nicht verarbeitet werden:\n\n{error_message}"
        )

    # ------------------------------------------------------------------ dialogs
    def _launch_authentication_center(self):
        """Opens the account status / credential dialog (AccountStatusCenterDialog(parent=None))."""
        dialog = AccountStatusCenterDialog(self)
        dialog.exec()

    def _launch_history_dialog(self):
        """Opens the chat history dialog (ChatHistoryDialog(database_connection, on_restore_callback, parent))."""
        dialog = ChatHistoryDialog(self.db, self._restore_session_context, self)
        dialog.exec()

    def _restore_session_context(self, session_id: int):
        """Slot called by the history dialog: reloads a stored session's last exchange."""
        self.active_session_id = session_id

        row = self.db.fetch_latest_log_for_session(session_id)
        if row is None:
            # Session exists but has no logged interaction (shouldn't
            # normally happen, since a session is only created right before
            # its first interaction, but handle it gracefully).
            self.last_prompt = ""
            self.last_response = ""
            self.last_diagnostic = None
            self.btn_export.setEnabled(False)
            self.prompt_editor.clear()
            self.response_viewer.clear()
            self.label_routing.setText("Zugeordneter Web-Agent: Keine Anfrage aktiv")
            self.label_status.setText(f"Status: Sitzung {session_id} (leer)")
            log.info(f"Dashboard context switched to empty session {session_id}.")
            return

        self.last_prompt = row["prompt_text"]
        self.last_response = row["response_text"]
        self.last_diagnostic = {
            "h_score": row["h_score"],
            "is_definitive_hallucination": row["h_score"] >= 0.6,  # display-bucket approximation; the exact
                                                                     # multi-condition flag isn't persisted in metrics_log
            "mean_drift_x": row["mean_drift_x"],
            "trace_covariance_y": row["trace_covariance_y"],
            "raw_metrics": {
                "sqd": row["sqd_val"],
                "rg_instability": row["rg_val"],
                "graph_entropy_drift": row["topo_drift"],
            },
        }

        self.prompt_editor.setPlainText(self.last_prompt)
        self.response_viewer.setText(self.last_response)
        self.btn_export.setEnabled(True)
        self.shield_widget.update_shield_state(self.last_diagnostic["h_score"], self.last_diagnostic["is_definitive_hallucination"])
        self.telemetry_panel.render_metrics_package(
            self.last_diagnostic["raw_metrics"], self.last_diagnostic["mean_drift_x"], self.last_diagnostic["trace_covariance_y"]
        )
        self.label_status.setText(f"Status: Sitzung {session_id} wiederhergestellt")
        log.info(f"Dashboard context restored from session {session_id}.")

    # ------------------------------------------------------------------- export
    def _execute_github_markdown_export(self):
        """Writes the last prompt, response and diagnostics to a Markdown file."""
        if not self.last_response:
            QMessageBox.information(self, "Nichts zu exportieren", "Es liegt noch keine Antwort vor.")
            return

        default_name = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        path, _ = QFileDialog.getSaveFileName(
            self, "Als GitHub Markdown exportieren", default_name, "Markdown (*.md)"
        )
        if not path:
            return

        diag = self.last_diagnostic or {}
        agent = (self.active_route_meta or {}).get("agent_name", "unbekannt")
        raw_json = json.dumps(diag.get("raw_metrics", {}), indent=2, ensure_ascii=False, default=str)

        lines = [
            "# AI Router & Hallucination Shield - Export",
            "",
            f"- **Zeitpunkt:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Web-Agent:** {agent}",
            f"- **Sitzung:** {self.active_session_id}",
            "",
            "## Prompt",
            "",
            self.last_prompt,
            "",
            "## Antwort",
            "",
            self.last_response,
            "",
            "## Diagnose",
            "",
            "| Metrik | Wert |",
            "|---|---|",
            f"| H-Score | {diag.get('h_score', 'n/a')} |",
            f"| Definitive Halluzination | {diag.get('is_definitive_hallucination', 'n/a')} |",
            f"| Mittlere Drift (x) | {diag.get('mean_drift_x', 'n/a')} |",
            f"| Spur der Kovarianz (y) | {diag.get('trace_covariance_y', 'n/a')} |",
            "",
            "<details>",
            "<summary>Rohmetriken</summary>",
            "",
            "```json",
            raw_json,
            "```",
            "",
            "</details>",
            "",
        ]

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.label_status.setText("Status: Export abgeschlossen")
            log.info(f"Markdown export written to {path}")
        except OSError as e:
            log.error(f"Markdown export failed: {e}")
            QMessageBox.critical(self, "Export fehlgeschlagen", str(e))

    # ----------------------------------------------------------------- shutdown
    def closeEvent(self, event):
        """Waits briefly for a running worker so the thread is not destroyed mid-run."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(3000)
        super().closeEvent(event)