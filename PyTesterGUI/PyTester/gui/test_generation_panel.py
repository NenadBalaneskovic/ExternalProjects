"""
TestGenerationPanel

This panel is responsible for:
- generating pytest test files based on the inference results
- using all test generation subsystems:
    • SmokeTestGenerator
    • TypeTestGenerator
    • BoundaryTestGenerator
    • PropertyTestGenerator
    • DocstringTestGenerator
    • TemplateRenderer

It displays:
- generated test code
- the number of tests created
- the target output file path

It writes test files into workspace/generated_tests.

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


class TestGenerationPanel(QWidget):
    """
    GUI panel for generating pytest tests from inferred structure and schema.
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

        self.generate_button = QPushButton("Generate Tests")
        self.generate_button.clicked.connect(self.generate_tests)
        button_row.addWidget(self.generate_button)

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
    # Test generation pipeline (multi‑file)
    # ------------------------------------------------------------
    def generate_tests(self) -> None:
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please select at least one file first.")
            return

        output_dir = Path(self.settings["paths"]["generated_tests"])
        output_dir.mkdir(parents=True, exist_ok=True)

        self.output_area.clear()

        for file_path in self.selected_files:
            registry = self.subsystems["structure_registry"]
            structure = registry.get_structure(file_path)

            if structure is None:
                QMessageBox.warning(
                    self,
                    "Not Analyzed",
                    f"The file '{file_path.name}' has not been inspected yet."
                )
                continue

            schema = self.subsystems["schema_builder"].get_schema(file_path)
            if schema is None:
                QMessageBox.warning(
                    self,
                    "No Inference",
                    f"Inference has not been run for '{file_path.name}'."
                )
                continue

            # Run test generators
            smoke_tests = self.subsystems["smoke_generator"].generate(
                file_path, structure, schema
            )
            type_tests = self.subsystems["type_tests_generator"].generate(
                file_path, structure, schema
            )
            boundary_tests = self.subsystems["boundary_tests_generator"].generate(
                file_path, structure, schema
            )
            property_tests = self.subsystems["property_tests_generator"].generate(
                file_path, structure, schema
            )
            docstring_tests = self.subsystems["docstring_tests_generator"].generate(
                file_path, structure, schema
            )

            # Render final test file
            renderer = self.subsystems["template_renderer"]
            final_test_code = renderer.render(
                smoke_tests,
                type_tests,
                boundary_tests,
                property_tests,
                docstring_tests
            )

            # Write test file
            test_filename = f"test_{file_path.stem}.py"
            test_path = output_dir / test_filename

            with open(test_path, "w", encoding="utf-8") as f:
                f.write(final_test_code)

            # Append to output area
            self.output_area.append(f"# === {file_path.name} ===")
            self.output_area.append(final_test_code)
            self.output_area.append("\n\n")

        QMessageBox.information(
            self,
            "Tests Generated",
            "Tests successfully generated for all selected files."
        )
