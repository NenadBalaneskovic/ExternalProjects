"""
gui/wheel_extractor_panel.py

GUI panel for:
- Searching PyPI for wheels for a given package+version.
- Filtering wheels by Python tag (e.g. cp312).
- Filtering wheels by platform (windows/linux/macos) based on OS profile.
- Displaying logs and results.
- Allowing manual or bulk wheel downloads.
- Clickable wheel links (open in browser).
"""

from __future__ import annotations

import os
import requests
import webbrowser

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

from ..core.wheel_extractor import WheelExtractor
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class WheelExtractorPanel(QWidget):
    """
    GUI panel for extracting wheels from PyPI.
    """

    PLATFORM_MAP = {
        "windows": "win_amd64",
        "linux": "manylinux",
        "macos": "macosx",
        "auto": None,
    }

    def __init__(self, project_state, parent=None) -> None:
        super().__init__(parent)
        self.project_state = project_state

        self.extractor = WheelExtractor()

        self._init_ui()
        self._apply_styles()

    # ------------------------------------------------------------------ #
    # UI initialization
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Input row
        input_row = QHBoxLayout()

        self.txt_package = QLineEdit()
        self.txt_package.setPlaceholderText("Package name (e.g. pandas)")

        self.txt_version = QLineEdit()
        self.txt_version.setPlaceholderText("Version (e.g. 3.0.1)")

        self.txt_python = QLineEdit()
        self.txt_python.setPlaceholderText("Python tag (e.g. cp312)")

        self.btn_find = QPushButton("Find Wheels")
        self.btn_find.clicked.connect(self._on_find_wheels)

        input_row.addWidget(QLabel("Package:"))
        input_row.addWidget(self.txt_package)
        input_row.addWidget(QLabel("Version:"))
        input_row.addWidget(self.txt_version)
        input_row.addWidget(QLabel("Python tag:"))
        input_row.addWidget(self.txt_python)
        input_row.addWidget(self.btn_find)

        layout.addLayout(input_row)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(120)
        layout.addWidget(self.log_output)

        # Results table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Package", "Version", "Wheel Link"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        # Make links clickable
        self.table.cellActivated.connect(self._on_cell_activated)
        self.table.cellDoubleClicked.connect(self._on_cell_activated)

        # Download all button
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self.btn_download_all = QPushButton("Download All Wheels")
        self.btn_download_all.clicked.connect(self._on_download_all)

        btn_row.addWidget(self.btn_download_all)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------ #
    # Styling
    # ------------------------------------------------------------------ #

    def _apply_styles(self) -> None:
        self.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                color: #e0e0e0;
            }
            QLineEdit {
                background-color: #2a2a2a;
                color: #e0e0e0;
                padding: 4px;
                border-radius: 4px;
            }
            QTextEdit {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border-radius: 4px;
            }
            QTableWidget {
                background-color: #2a2a2a;
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
    # Logging helper
    # ------------------------------------------------------------------ #

    def _log(self, msg: str) -> None:
        self.log_output.append(msg)
        self.log_output.moveCursor(QTextCursor.End)

    # ------------------------------------------------------------------ #
    # Clickable wheel links
    # ------------------------------------------------------------------ #

    def _on_cell_activated(self, row: int, col: int) -> None:
        if col != 2:
            return

        url = self.table.item(row, col).text()
        if url:
            webbrowser.open(url)
            self._log(f"[INFO] Opening wheel link in browser: {url}")

    # ------------------------------------------------------------------ #
    # Find wheels
    # ------------------------------------------------------------------ #

    def _on_find_wheels(self) -> None:
        package = self.txt_package.text().strip()
        version = self.txt_version.text().strip()
        python_tag = self.txt_python.text().strip() or None

        # Determine platform tag from OS profile
        os_profile = self.project_state.os_profile.lower()
        platform_tag = self.PLATFORM_MAP.get(os_profile)

        if not package or not version:
            self._log("[ERR] Please enter package and version.")
            return

        self._log(f"[INFO] Searching wheels for {package}=={version}...")
        if python_tag:
            self._log(f"[INFO] Python tag filter: {python_tag}")
        if platform_tag:
            self._log(f"[INFO] Platform filter: {platform_tag}")

        wheels, logs = self.extractor.find_wheels(
            package,
            version,
            python_tag=python_tag,
            platform_tag=platform_tag,
        )

        for line in logs:
            if line.lower().startswith("failure"):
                self._log(f"[ERR] {line}")
            elif line.lower().startswith("success"):
                self._log(f"[INFO] {line}")
            else:
                self._log(f"[INFO] {line}")

        if not wheels:
            self._log("[ERR] No wheels found.")
            return

        # Add rows to table
        for wheel in wheels:
            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, QTableWidgetItem(wheel.package))
            self.table.setItem(row, 1, QTableWidgetItem(wheel.version))

            link_item = QTableWidgetItem(wheel.url)
            link_item.setForeground(Qt.cyan)
            link_item.setToolTip("Open wheel link in browser")
            link_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 2, link_item)

        self._log(f"[INFO] Added {len(wheels)} wheels to table.")

    # ------------------------------------------------------------------ #
    # Download all wheels
    # ------------------------------------------------------------------ #

    def _on_download_all(self) -> None:
        if self.table.rowCount() == 0:
            self._log("[ERR] No wheels to download.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Download Folder")
        if not folder:
            return

        count = 0

        for row in range(self.table.rowCount()):
            url = self.table.item(row, 2).text()
            filename = url.split("/")[-1]
            dest = os.path.join(folder, filename)

            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()

                with open(dest, "wb") as f:
                    f.write(resp.content)

                count += 1
                self._log(f"[INFO] Downloaded: {filename}")

            except Exception as exc:
                self._log(f"[ERR] Failed to download {filename}: {exc}")

        self._log(f"[INFO] Download complete. {count} wheels downloaded.")
