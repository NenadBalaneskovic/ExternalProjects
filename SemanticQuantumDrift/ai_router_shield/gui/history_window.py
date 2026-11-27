# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  gui/history_window.py
ROLE:    Interactive historical session browser dialog presenting past web-threads,
         calculated metadata, and localized phase-space evaluation charts.
================================================================================
"""

import logging
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, 
                             QTableWidget, QTableWidgetItem, QPushButton, 
                             QHeaderView, QMessageBox, QAbstractItemView)
from PyQt6.QtCore import Qt

from gui.plot_window import MetricsPlotDialog

log = logging.getLogger("HistoryWindowEngine")


class ChatHistoryDialog(QDialog):
    """
    A modeless-styled modal dialog pulling operational data stacks straight out 
    of the local SQLite persistence layer to restore context-cached threads.
    """
    def __init__(self, database_connection, on_restore_callback, parent=None):
        super().__init__(parent)
        self.db = database_connection
        self.restore_callback = on_restore_callback
        
        self.setWindowTitle("CHAT-VERLAUF & SITZUNGS-MANAGEMENT")
        self.resize(900, 520)
        self.setModal(True)
        
        # Cache array to track un-filtered dataset rows during search queries
        self.cached_sessions = []
        
        self._build_history_interface()
        self.load_history_from_storage()

    def _build_history_interface(self):
        """Constructs high-contrast, scalable data grid layouts and event attachments."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # 1. Search & Filter Bar Component
        self.search_filter = QLineEdit()
        self.search_filter.setPlaceholderText("Suche/Filter: z.B. Quantenmechanik, Paradoxon, Claude...")
        self.search_filter.setStyleSheet("""
            padding: 8px; 
            font-size: 13px; 
            background-color: #161b22; 
            color: #c9d1d9; 
            border: 1px solid #30363d; 
            border-radius: 6px;
        """)
        self.search_filter.textChanged.connect(self._execute_live_filtering)
        layout.addWidget(self.search_filter)
        
        # 2. Main History Tabular Grid Structure
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "Thema / Prompt-Auszug", "KI-Agent", "Typ", "H-Score"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        # Multi-row selection (Ctrl/Shift-click) stands in for per-row
        # checkboxes here: it gives the same bulk-select capability with a
        # much simpler, less error-prone implementation in a QTableWidget.
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        
        # Apply strict layout rendering constraints to headers
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        
        self.table.setStyleSheet("""
            QTableWidget { background-color: #0d1117; color: #c9d1d9; gridline-color: #30363d; border: 1px solid #30363d; }
            QTableWidget::item:selected { background-color: #21262d; color: #58a6ff; }
            QHeaderView::section { background-color: #161b22; color: #8b949e; border: 1px solid #30363d; padding: 5px; font-weight: bold; }
        """)
        layout.addWidget(self.table)

        hint = QLineEdit()
        hint.setReadOnly(True)
        hint.setText("Tipp: Strg-Klick / Umschalt-Klick für Mehrfachauswahl zum Bulk-Löschen.")
        hint.setStyleSheet("border: none; background: transparent; color: #8b949e; font-style: italic; font-size: 11px;")
        layout.addWidget(hint)
        
        # 3. Action Control Layout (Horizontal Command Block)
        btn_layout = QHBoxLayout()
        
        self.btn_restore = QPushButton("Chat im Hauptfenster fortsetzen")
        self.btn_plot = QPushButton("Metrik-Plot zeigen")
        self.btn_delete_selected = QPushButton("Ausgewählte löschen")
        self.btn_delete_all = QPushButton("Gesamten Verlauf löschen")
        self.btn_close = QPushButton("Schließen")

        self.btn_delete_selected.setStyleSheet("color: #f85149;")
        self.btn_delete_all.setStyleSheet("color: #f85149; font-weight: bold;")
        
        btn_layout.addWidget(self.btn_restore)
        btn_layout.addWidget(self.btn_plot)
        btn_layout.addWidget(self.btn_delete_selected)
        btn_layout.addWidget(self.btn_delete_all)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)
        
        # Bind tactical signal connections to backend functional loops
        self.btn_close.clicked.connect(self.reject)
        self.btn_restore.clicked.connect(self._process_session_restoration)
        self.btn_plot.clicked.connect(self._render_trajectory_visualization)
        self.btn_delete_selected.clicked.connect(self._delete_selected_sessions)
        self.btn_delete_all.clicked.connect(self._delete_entire_history)

    def load_history_from_storage(self):
        """Pulls comprehensive historical session arrays straight out of database matrices."""
        try:
            self.cached_sessions = self.db.fetch_all_active_sessions()
            self._render_table_rows(self.cached_sessions)
        except Exception as e:
            log.error(f"Persistence audit failure inside history window: {str(e)}")
            QMessageBox.critical(self, "Database Error", f"Fehler beim Laden des Verlaufs: {str(e)}")

    def _render_table_rows(self, dataset: list):
        """Populates individual cells dynamically with type signatures and telemetry alerts."""
        self.table.setRowCount(0)
        
        for row_idx, session in enumerate(dataset):
            self.table.insertRow(row_idx)
            
            # Column 0: Standardized Padded Primary Key ID
            id_item = QTableWidgetItem(f"#{session['id']:03d}")
            id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Column 1: Abstract Extract Title String
            title_item = QTableWidgetItem(session['title'])
            
            # Column 2: Selected Computing Architecture Node
            agent_item = QTableWidgetItem(session['agent_name'].split()[0]) # Short identifier
            
            # Column 3: Relational Protocol Pathway Type
            type_item = QTableWidgetItem(session['chat_type'])
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Column 4: Highly Scannable Hyperbolic Evaluation Score Matrix
            h_val = session['last_h_score']
            score_str = f"{h_val:.2f}"
            
            # Inject telemetric warning vector emoji explicitly on parameter breach condition
            if h_val >= 0.6:
                score_str += " 🚨"
                
            score_item = QTableWidgetItem(score_str)
            score_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Append rows safely into the QTableWidget stack
            self.table.setItem(row_idx, 0, id_item)
            self.table.setItem(row_idx, 1, title_item)
            self.table.setItem(row_idx, 2, agent_item)
            self.table.setItem(row_idx, 3, type_item)
            self.table.setItem(row_idx, 4, score_item)

    def _execute_live_filtering(self, filter_text: str):
        """Filters row records reactively across multiple fields via linear string scans."""
        query = filter_text.strip().lower()
        if not query:
            self._render_table_rows(self.cached_sessions)
            return
            
        filtered_set = []
        for sess in self.cached_sessions:
            if (query in str(sess['id']).lower() or 
                query in sess['title'].lower() or 
                query in sess['agent_name'].lower() or 
                query in sess['chat_type'].lower()):
                filtered_set.append(sess)
                
        self._render_table_rows(filtered_set)

    def _selected_session_ids(self) -> list:
        """Returns the database IDs of every currently highlighted row (0 to many)."""
        ids = []
        for item in self.table.selectedItems():
            if item.column() == 0:
                ids.append(int(item.text().replace("#", "")))
        return ids

    def _process_session_restoration(self):
        """Passes the authorized primary key index back to core main thread handlers."""
        selected_row = self.table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(self, "Auswahl erforderlich", "Bitte wählen Sie eine Sitzung aus der Tabelle aus.")
            return
            
        # Isolate the numeric integer token from the string item cell
        raw_id_str = self.table.item(selected_row, 0).text().replace("#", "")
        session_database_id = int(raw_id_str)
        
        log.info(f"Triggering pipeline context restoration for Session Primary Key: [{session_database_id}]")
        self.restore_callback(session_id=session_database_id)
        self.accept()

    def _render_trajectory_visualization(self):
        """Pulls the session's full metrics_log trajectory and renders it via Matplotlib."""
        selected_row = self.table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(self, "Auswahl erforderlich", "Bitte wählen Sie eine Sitzung aus, um den Metrik-Plot anzuzeigen.")
            return

        raw_id_str = self.table.item(selected_row, 0).text().replace("#", "")
        session_database_id = int(raw_id_str)
        session_title = self.table.item(selected_row, 1).text()

        try:
            rows = self.db.fetch_metrics_for_session(session_database_id)
        except Exception as e:
            log.error(f"Failed to fetch trajectory data for session {session_database_id}: {str(e)}")
            QMessageBox.critical(self, "Database Error", f"Fehler beim Laden der Trajektorie: {str(e)}")
            return

        dialog = MetricsPlotDialog(session_title, rows, self)
        dialog.exec()

    def _delete_selected_sessions(self):
        """Bulk-deletes every currently highlighted session (and its logged steps)."""
        ids = self._selected_session_ids()
        if not ids:
            QMessageBox.warning(self, "Auswahl erforderlich", "Bitte markieren Sie mindestens eine Sitzung (Strg/Umschalt-Klick für mehrere).")
            return

        confirm = QMessageBox.question(
            self, "Löschen bestätigen",
            f"{len(ids)} Sitzung(en) inkl. aller zugehörigen Metrik-Daten unwiderruflich löschen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            self.db.delete_sessions(ids)
        except Exception as e:
            log.error(f"Failed to delete sessions {ids}: {str(e)}")
            QMessageBox.critical(self, "Database Error", f"Löschen fehlgeschlagen: {str(e)}")
            return

        self.load_history_from_storage()

    def _delete_entire_history(self):
        """Wipes every session and every logged trajectory step after confirmation."""
        confirm = QMessageBox.question(
            self, "Gesamten Verlauf löschen",
            "ALLE Sitzungen und Metrik-Verläufe unwiderruflich löschen? Dies kann nicht rückgängig gemacht werden.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            self.db.delete_all_sessions()
        except Exception as e:
            log.error(f"Failed to wipe history: {str(e)}")
            QMessageBox.critical(self, "Database Error", f"Löschen fehlgeschlagen: {str(e)}")
            return

        self.load_history_from_storage()


log.info("Production gui/history_window.py component matrix initialized.")