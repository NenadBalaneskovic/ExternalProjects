"""
SmokeTestGenerator

This subsystem is responsible for:
- generating minimal smoke tests for each symbol in the schema
- verifying importability and safe invocation
- producing pytest-compatible test files
- ensuring deterministic, side-effect-free test generation

It is intentionally conservative:
no arbitrary execution, no unsafe calls, no mutation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

from core.utils import indent


class SmokeTestGenerator:
    """
    Generate smoke tests from a canonical schema.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.allow_zero_arg_calls = settings["testgen"]["smoke"]["allow_zero_arg_calls"]

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def generate(self, file_path: Path, structure: Dict[str, Any], schema: Dict[str, Any]) -> str:
        module_name = file_path.stem

        lines: List[str] = []
        lines.append("import pytest")

        # 🔥 WICHTIG: Source‑Modul korrekt importieren
        lines.append(f"from workspace.source import {module_name}")
        lines.append("")

        for name, info in schema.items():
            lines.extend(self._generate_test_case(module_name, name, info))
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------
    # Individual test case generation
    # ------------------------------------------------------------
    def _generate_test_case(self, module_name: str, name: str, info: Dict[str, Any]) -> List[str]:
        """
        Generate a single smoke test case.

        Corrected rules:
        - NEVER instantiate classes.
        - NEVER call methods.
        - ONLY verify importability and callability.
        """

        test_name = f"test_smoke_{name.replace('.', '_')}"
        lines: List[str] = [f"def {test_name}():"]

        # ------------------------------------------------------------
        # Case 1: class-level symbol
        # ------------------------------------------------------------
        if "." not in name:
            cls_name = name

            # 🔥 Source‑Modul korrekt referenzieren
            lines.append(indent(f"cls = {module_name}.{cls_name}", 4))
            lines.append(indent("assert callable(cls)", 4))
            lines.append(indent("# class import verified; instantiation skipped", 4))
            return lines

        # ------------------------------------------------------------
        # Case 2: method-level symbol
        # ------------------------------------------------------------
        cls_name, method_name = name.split(".")

        # 🔥 Source‑Modul korrekt referenzieren
        lines.append(indent(f"cls = {module_name}.{cls_name}", 4))
        lines.append(indent("assert callable(cls)", 4))
        lines.append(indent(f"assert hasattr(cls, '{method_name}')", 4))
        lines.append(indent("# method existence verified; invocation skipped", 4))

        return lines
