"""
InferencePanel

This panel is responsible for:
- running the full inference pipeline on the uploaded Python file
- static analysis
- semantic analysis
- dynamic probing
- type fusion
- schema building

It displays:
- inferred types
- inferred shapes
- inferred behaviors
- merged schema

It reads the structure stored in StructureRegistry and uses all
inference subsystems initialized in run.py.

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


class InferencePanel(QWidget):
    """
    GUI panel for performing type and behavior inference on uploaded Python files.
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

        self.file_label = QLabel("No files selected.")
        layout.addWidget(self.file_label)

        button_row = QHBoxLayout()
        layout.addLayout(button_row)

        self.select_button = QPushButton("Select Uploaded Files")
        self.select_button.clicked.connect(self.select_files)
        button_row.addWidget(self.select_button)

        self.infer_button = QPushButton("Run Inference")
        self.infer_button.clicked.connect(self.run_inference)
        button_row.addWidget(self.infer_button)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(self.output_area)

    # ------------------------------------------------------------
    # Multi‑Select File selection
    # ------------------------------------------------------------
    def select_files(self) -> None:
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
    # Inference pipeline (multi‑file)
    # ------------------------------------------------------------
    def run_inference(self) -> None:
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please select at least one file first.")
            return

        self.output_area.clear()

        registry = self.subsystems["structure_registry"]

        for file_path in self.selected_files:
            structure = registry.get_structure(file_path)

            if structure is None:
                self.output_area.append(f"# {file_path.name}: NOT ANALYZED\n")
                continue

            # Run inference subsystems
            static = self.subsystems["static_analyzer"].analyze(structure)
            semantic = self.subsystems["semantic_analyzer"].analyze(structure)
            dynamic = self.subsystems["dynamic_probe"].probe(structure)
            fused = self.subsystems["type_fusion"].merge(static, semantic, dynamic)

            # SchemaBuilder requires file_path + fused_info
            schema = self.subsystems["schema_builder"].build(
                file_path=file_path,
                fused_info=fused
            )

            # Append results for this file
            text_output = self._format_inference(static, semantic, dynamic, fused, schema)

            self.output_area.append(f"# === {file_path.name} ===\n")
            self.output_area.append(text_output)
            self.output_area.append("\n\n")

        QMessageBox.information(self, "Inference Complete", "Inference executed for all selected files.")

    # ------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------
    def _format_inference(
        self,
        static: Dict[str, Any],
        semantic: Dict[str, Any],
        dynamic: Dict[str, Any],
        fused: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> str:

        lines: list[str] = []
        lines.append("=== Inference Results ===\n")

        lines.append("Static Analysis:")
        for name, info in static.items():
            lines.append(f"  - {name}: {info}")
        lines.append("")

        lines.append("Semantic Analysis:")
        for name, info in semantic.items():
            lines.append(f"  - {name}: {info}")
        lines.append("")

        lines.append("Dynamic Probe:")
        for name, info in dynamic.items():
            lines.append(f"  - {name}: {info}")
        lines.append("")

        lines.append("Type Fusion:")
        for name, info in fused.items():
            lines.append(f"  - {name}: {info}")
        lines.append("")

        lines.append("Schema:")
        for name, info in schema.items():
            lines.append(f"  - {name}: {info}")
        lines.append("")

        return "\n".join(lines)
