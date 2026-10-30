"""
gui/top_bar.py

Top bar widget for PyDependencyInspector.

Updated to:
- Remove PyInstaller terminology
- Treat "Build" as "Collect Metadata"
- Keep Scan, Build, Export actions
- Apply dark graphite + cyan styling
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
)
from PySide6.QtCore import Qt, Signal

from ..config.paths import config


class TopBar(QWidget):
    """
    Balanced top bar layout:

    ┌──────────────────────────────────────────────────────────────────────┐
    │ [OS Profile ▼]  [Scan]  [Collect Metadata]  [Export]   Project Name  │
    └──────────────────────────────────────────────────────────────────────┘

    Signals:
    - projectRenamed(str)
    - osProfileChanged(str)
    - runResolutionRequested()
    - runBuildRequested()        ← now means "collect metadata"
    - exportRequested()
    """

    projectRenamed = Signal(str)
    osProfileChanged = Signal(str)
    runResolutionRequested = Signal()
    runBuildRequested = Signal()
    exportRequested = Signal()

    def __init__(self, project_state) -> None:
        super().__init__()
        self.project_state = project_state

        self._init_ui()
        self._apply_styles()

    # ------------------------------------------------------------------ #
    # UI initialization
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)

        # OS profile selector
        self.os_combo = QComboBox()
        self.os_combo.addItems(["auto", "windows", "linux", "macos"])
        self.os_combo.setCurrentText(self.project_state.os_profile)
        self.os_combo.currentTextChanged.connect(self.osProfileChanged)

        # Scan dependencies
        self.btn_scan = QPushButton("Scan")
        self.btn_scan.clicked.connect(self.runResolutionRequested)

        # Collect metadata (formerly "Build")
        self.btn_build = QPushButton("Collect Metadata")
        self.btn_build.clicked.connect(self.runBuildRequested)

        # Export (requirements.txt only)
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self.exportRequested)

        # Project name editor
        self.project_edit = QLineEdit(self.project_state.project_name)
        self.project_edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.project_edit.textChanged.connect(self.projectRenamed)

        # Layout assembly
        layout.addWidget(self.os_combo)
        layout.addWidget(self.btn_scan)
        layout.addWidget(self.btn_build)
        layout.addWidget(self.btn_export)

        layout.addStretch(1)
        layout.addWidget(self.project_edit)

    # ------------------------------------------------------------------ #
    # Styling
    # ------------------------------------------------------------------ #

    def _apply_styles(self) -> None:
        """Apply dark graphite + cyan theme."""
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
            }
            QComboBox {
                background-color: #2b2b2b;
                color: #e0e0e0;
                padding: 4px 8px;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
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
            QLineEdit {
                background-color: #2b2b2b;
                color: #e0e0e0;
                padding: 6px 10px;
                border: 1px solid #444;
                border-radius: 4px;
                min-width: 260px;
                font-weight: 600;
            }
        """)
