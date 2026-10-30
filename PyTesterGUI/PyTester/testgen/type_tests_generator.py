"""
TypeTestsGenerator

Corrected to:
- NEVER instantiate classes
- NEVER call methods
- ONLY run runtime checks for free functions when safe and allowed
- Remain deterministic and side‑effect‑free
- ALWAYS import the correct source module for coverage
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

from core.utils import indent


class TypeTestsGenerator:
    """
    Generate type-oriented tests from a canonical schema.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.enable_runtime_checks = settings["testgen"]["type"]["enable_runtime_checks"]

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
            if info.get("kind") not in ("function", "method"):
                continue

            lines.extend(self._generate_test_case(module_name, name, info))
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------
    # Individual test case generation
    # ------------------------------------------------------------
    def _generate_test_case(self, module_name: str, name: str, info: Dict[str, Any]) -> List[str]:
        """
        Generate a single type test case.

        Final rules:
        - Free functions: optional runtime type checks.
        - Class methods: ONLY verify existence and callability, no instantiation, no calls.
        """

        test_name = f"test_types_{name.replace('.', '_')}"
        lines: List[str] = [f"def {test_name}():"]

        # ------------------------------------------------------------
        # Case 1: free function
        # ------------------------------------------------------------
        if "." not in name:
            func_name = name

            # 🔥 Source‑Modul korrekt referenzieren
            lines.append(indent(f"func = {module_name}.{func_name}", 4))
            lines.append(indent("assert callable(func)", 4))

            if self.enable_runtime_checks:
                arg_values = self._build_dummy_args(info.get("args", {}))

                if arg_values is None:
                    lines.append(indent("# cannot build dummy args → skip runtime checks", 4))
                    return lines

                call = f"func()" if arg_values == "" else f"func({arg_values})"
                lines.append(indent(f"result = {call}", 4))

                expected_type = self._expected_runtime_type(info.get("return"))
                if expected_type:
                    lines.append(indent(f"assert isinstance(result, {expected_type})", 4))

            return lines

        # ------------------------------------------------------------
        # Case 2: class method (no instantiation, no calls)
        # ------------------------------------------------------------
        cls_name, method_name = name.split(".")

        # 🔥 Source‑Modul korrekt referenzieren
        lines.append(indent(f"cls = {module_name}.{cls_name}", 4))
        lines.append(indent("assert callable(cls)", 4))
        lines.append(indent(f"assert hasattr(cls, '{method_name}')", 4))
        lines.append(indent("# type tests for methods: existence and callability only; instantiation skipped", 4))

        return lines

    # ------------------------------------------------------------
    # Dummy argument construction
    # ------------------------------------------------------------
    def _build_dummy_args(self, args: Dict[str, Optional[str]]) -> Optional[str]:
        if not args:
            return ""

        exprs: List[str] = []
        for name, ann in args.items():
            exprs.append(self._dummy_value_for_annotation(ann))

        return ", ".join(exprs) if exprs else None

    def _dummy_value_for_annotation(self, ann: Optional[str]) -> str:
        if not ann:
            return "None"

        lowered = ann.lower()

        if lowered in ("int", "float", "complex"):
            return "0"
        if lowered == "bool":
            return "False"
        if lowered == "str":
            return "''"
        if "list" in lowered or "tuple" in lowered or "set" in lowered:
            return "[]"
        if "dict" in lowered or "mapping" in lowered:
            return "{}"
        return "None"

    # ------------------------------------------------------------
    # Expected runtime type mapping
    # ------------------------------------------------------------
    def _expected_runtime_type(self, ann: Optional[str]) -> Optional[str]:
        if not ann:
            return None

        lowered = ann.lower()

        if lowered in ("int", "float", "complex"):
            return "(int, float, complex)"
        if lowered == "bool":
            return "bool"
        if lowered == "str":
            return "str"
        if "list" in lowered:
            return "list"
        if "tuple" in lowered:
            return "tuple"
        if "set" in lowered:
            return "set"
        if "dict" in lowered or "mapping" in lowered:
            return "dict"

        return None
