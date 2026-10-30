"""
TemplateRenderer

Corrected to:
- normalize empty blocks safely
- preserve deterministic ordering
- remain pure and side‑effect‑free
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from core.utils import indent


class TemplateRenderer:
    """
    Combine multiple test sections into a single pytest file.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.indent_spaces = settings["testgen"]["renderer"]["indent_spaces"]

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def render(
        self,
        smoke_tests: str,
        type_tests: str,
        boundary_tests: str,
        property_tests: str,
        docstring_tests: str,
    ) -> str:
        """
        Combine all test sections into one final test file.
        """

        def normalize_block(block: str) -> str:
            text = block.strip()
            return text if text else "# (no tests generated)"

        parts = [
            "# === Smoke Tests ===",
            normalize_block(smoke_tests),
            "",
            "# === Type Tests ===",
            normalize_block(type_tests),
            "",
            "# === Boundary Tests ===",
            normalize_block(boundary_tests),
            "",
            "# === Property Tests ===",
            normalize_block(property_tests),
            "",
            "# === Docstring Tests ===",
            normalize_block(docstring_tests),
        ]

        return "\n".join(parts)

    # ------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------
    def wrap_test_case(self, name: str, body: str) -> str:
        indented_body = indent(body, self.indent_spaces)
        return f"def {name}():\n{indented_body}\n"

    def render_section(self, title: str, content: str) -> str:
        cleaned = content.strip()
        return f"# {title}\n{cleaned}"

    def join(self, *parts: str) -> str:
        cleaned = [p.strip() for p in parts if p.strip()]
        return "\n\n".join(cleaned)

    def normalize(self, text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(lines)
