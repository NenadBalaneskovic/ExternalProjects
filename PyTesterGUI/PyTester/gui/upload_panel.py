"""
UploadPanel

This panel is responsible for:
- letting the user select a Python file (.py)
- copying the file into workspace/uploaded_files
- triggering syntax checking and AST inspection
- notifying downstream panels that a new file is available

It is the first step in the PyTester workflow.
Minimal erweitert für:
- Multi‑Select (getOpenFileNames)
- Schleife über mehrere Dateien
- Auto‑Copy nach workspace/source/
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox
)


class UploadPanel(QWidget):
    """
    GUI panel for uploading Python files into the PyTester workspace.
    """

    def __init__(self, settings: Dict[str, Any], subsystems: Dict[str, Any]) -> None:
        super().__init__()

        self.settings = settings
        self.subsystems = subsystems

        # Workspace paths
        self.workspace_dir = Path(self.settings["paths"]["workspace"])
        self.upload_dir = Path(self.settings["paths"]["uploaded_files"])
        self.source_dir = Path(self.settings["paths"]["source"])

        # Last selected files (now a list)
        self.selected_files: List[Path] = []

        # ------------------------------------------------------------
        # GUI Layout
        # ------------------------------------------------------------
        layout = QVBoxLayout()
        self.setLayout(layout)

        # File selection label
        self.file_label = QLabel("No files selected.")
        layout.addWidget(self.file_label)

        # Buttons row
        button_row = QHBoxLayout()
        layout.addLayout(button_row)

        # Select file(s) button
        self.select_button = QPushButton("Select Python Files")
        self.select_button.clicked.connect(self.select_files)
        button_row.addWidget(self.select_button)

        # Load file(s) button
        self.load_button = QPushButton("Load Files")
        self.load_button.clicked.connect(self.load_files)
        button_row.addWidget(self.load_button)

    # ------------------------------------------------------------
    # Multi‑Select File selection
    # ------------------------------------------------------------
    def select_files(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Python Files",
            "",
            "Python Files (*.py)"
        )

        if file_paths:
            self.selected_files = [Path(p) for p in file_paths]
            names = ", ".join([p.name for p in self.selected_files])
            self.file_label.setText(f"Selected: {names}")

    # ------------------------------------------------------------
    # Load + validate multiple files
    # ------------------------------------------------------------
    def load_files(self) -> None:
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please select at least one Python file first.")
            return

        # Ensure workspace/uploaded_files exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir.mkdir(parents=True, exist_ok=True)

        for file_path in self.selected_files:
            # Copy into workspace/uploaded_files
            target_path = self.upload_dir / file_path.name
            shutil.copy(file_path, target_path)

            # Auto‑copy into workspace/source (Coverage needs this)
            shutil.copy(file_path, self.source_dir / file_path.name)

            # Run syntax checker
            syntax_ok = self.subsystems["syntax_checker"].check_file(target_path)
            if not syntax_ok:
                QMessageBox.critical(self, "Syntax Error",
                                     f"The file '{file_path.name}' contains syntax errors.")
                continue

            # Run AST inspector
            structure = self.subsystems["ast_inspector"].inspect_file(target_path)

            # Register structure for downstream panels
            self.subsystems["structure_registry"].store_structure(target_path, structure)

        QMessageBox.information(self, "Files Loaded",
                                "All selected files were successfully loaded and analyzed.")

