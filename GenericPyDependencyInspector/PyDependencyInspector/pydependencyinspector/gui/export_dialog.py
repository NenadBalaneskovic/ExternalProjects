"""
gui/export_dialog.py

Simplified: only export requirements.txt
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QPushButton,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt

from ..core.requirements_exporter import RequirementsExporter
from ..utils.logging_utils import get_logger
from ..config.paths import config

logger = get_logger(__name__)


class ExportDialog(QDialog):
    """
    Modal dialog for selecting export options.
    Only requirements.txt is supported now.
    """

    def __init__(self, project_state, parent=None) -> None:
        super().__init__(parent)
        self.project_state = project_state

        self.setWindowTitle("Export")
        self.setModal(True)
        self.resize(420, 180)

        self._init_ui()
        self._apply_styles()

    # ------------------------------------------------------------------ #
    # UI initialization
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.chk_requirements = QCheckBox("requirements.txt")
        layout.addWidget(self.chk_requirements)

        layout.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self._on_export)

        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_export)

        layout.addLayout(btn_row)

    # ------------------------------------------------------------------ #
    # Styling
    # ------------------------------------------------------------------ #

    def _apply_styles(self) -> None:
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
                color: #e0e0e0;
            }
            QCheckBox {
                font-size: 14px;
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                padding: 6px 14px;
                border-radius: 4px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0b4f75;
            }
        """)

    # ------------------------------------------------------------------ #
    # Export logic
    # ------------------------------------------------------------------ #

    def _on_export(self) -> None:
        if not self.chk_requirements.isChecked():
            QMessageBox.warning(self, "No Selection", "Please select requirements.txt to export.")
            return

        export_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            config.project_report_dir,
        )
        if not export_dir:
            return

        ok = self._export_requirements(export_dir)

        QMessageBox.information(
            self,
            "Export Complete",
            "requirements.txt exported successfully." if ok else "Export failed."
        )

        self.accept()

    # ------------------------------------------------------------------ #
    # requirements.txt export
    # ------------------------------------------------------------------ #

    def _export_requirements(self, export_dir: str) -> bool:
        if not self.project_state.resolution_result:
            logger.error("Cannot export requirements: no resolution available.")
            return False

        path = os.path.join(export_dir, "requirements.txt")
        exporter = RequirementsExporter()
        result = exporter.export(
            self.project_state.resolution_result.flat_list,
            path,
        )
        return result.success
