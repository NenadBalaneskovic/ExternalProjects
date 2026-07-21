"""
InspectionPanel

This panel is responsible for:
- displaying the AST structure of the uploaded Python file
- showing extracted docstrings and type annotations
- providing a human‑readable overview of classes, functions, and methods
- serving as the bridge between upload and inference

It reads the structure stored in StructureRegistry by UploadPanel.

Minimal erweitert für:
- Multi‑Select (getOpenFileNames)
- Schleife über mehrere Dateien
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QMessageBox,
    QFileDialog
)


class InspectionPanel(QWidget):
    """
    GUI panel for inspecting the structure of uploaded Python files.
    """

    def __init__(self, settings: Dict[str, Any], subsystems: Dict[str, Any]) -> None:
        super().__init__()

        self.settings = settings
        self.subsystems = subsystems

        # Multi‑select: now a list
        self.selected_files: List[Path] = []

        # ------------------------------------------------------------
        # GUI Layout
        # ------------------------------------------------------------
        layout = QVBoxLayout()
        self.setLayout(layout)

        # File label
        self.file_label = QLabel("No files selected.")
        layout.addWidget(self.file_label)

        # Buttons row
        button_row = QHBoxLayout()
        layout.addLayout(button_row)

        # Select files button
        self.select_button = QPushButton("Select Uploaded Files")
        self.select_button.clicked.connect(self.select_files)
        button_row.addWidget(self.select_button)

        # Inspect button
        self.inspect_button = QPushButton("Inspect Files")
        self.inspect_button.clicked.connect(self.inspect_files)
        button_row.addWidget(self.inspect_button)

        # Text area for displaying structure
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(self.output_area)

    # ------------------------------------------------------------
    # Multi‑Select File selection
    # ------------------------------------------------------------
    def select_files(self) -> None:
        """
        Select multiple files from workspace/uploaded_files for inspection.
        """
        upload_dir = Path(self.settings["paths"]["uploaded_files"])

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Uploaded Python Files",
            str(upload_dir),
            "Python Files (*.py)"
        )

        if file_paths:
            self.selected_files = [Path(p) for p in file_paths]
            names = ", ".join([p.name for p in self.selected_files])
            self.file_label.setText(f"Selected: {names}")

    # ------------------------------------------------------------
    # Inspection logic (multi‑file)
    # ------------------------------------------------------------
    def inspect_files(self) -> None:
        """
        Retrieve stored AST structures for all selected files and display them.
        """
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please select at least one file first.")
            return

        registry = self.subsystems["structure_registry"]
        self.output_area.clear()

        for file_path in self.selected_files:
            structure = registry.get_structure(file_path)

            if structure is None:
                self.output_area.append(f"# {file_path.name}: NOT ANALYZED\n")
                continue

            text_output = self._format_structure(structure)

            self.output_area.append(f"# === {file_path.name} ===\n")
            self.output_area.append(text_output)
            self.output_area.append("\n\n")

        QMessageBox.information(
            self,
            "Inspection Complete",
            "Inspection executed for all selected files."
        )

    # ------------------------------------------------------------
    # Structure formatting
    # ------------------------------------------------------------
    def _format_structure(self, structure: Dict[str, Any]) -> str:
        """
        Convert the structure dictionary into a readable text block.
        """
        lines: list[str] = []
        lines.append("=== AST Structure ===\n")

        # Classes
        classes = structure.get("classes", {})
        if classes:
            lines.append("Classes:")
            for cls_name, cls_info in classes.items():
                lines.append(f"  - {cls_name}")
                methods = cls_info.get("methods", [])
                for m in methods:
                    lines.append(f"      • method: {m}")
            lines.append("")

        # Functions
        functions = structure.get("functions", [])
        if functions:
            lines.append("Functions:")
            for func in functions:
                lines.append(f"  - {func}")
            lines.append("")

        # Docstrings
        docstrings = structure.get("docstrings", {})
        if docstrings:
            lines.append("Docstrings:")
            for name, doc in docstrings.items():
                lines.append(f"  - {name}:")
                lines.append(f"      {doc.strip()}")
            lines.append("")

        # Annotations
        annotations = structure.get("annotations", {})
        if annotations:
            lines.append("Type Annotations:")
            for name, ann in annotations.items():
                lines.append(f"  - {name}: {ann}")
            lines.append("")

        return "\n".join(lines)
