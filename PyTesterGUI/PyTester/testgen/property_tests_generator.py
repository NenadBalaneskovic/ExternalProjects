"""
PropertyTestsGenerator

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


class PropertyTestsGenerator:
    """
    Generate property-based tests from a canonical schema.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.enable_runtime_checks = settings["testgen"]["property"]["enable_runtime_checks"]

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

            props = self._infer_properties(info)
            if not props:
                continue

            lines.extend(self._generate_test_case(module_name, name, info, props))
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------
    # Property inference
    # ------------------------------------------------------------
    def _infer_properties(self, info: Dict[str, Any]) -> List[str]:
        intent = info.get("intent")
        props: List[str] = []

        if intent == "validation":
            props.append("no_crash_on_valid_input")

        if intent == "computation":
            props.append("deterministic_output")

        if intent == "transformation":
            props.append("output_type_consistent")

        return props

    # ------------------------------------------------------------
    # Individual test case generation
    # ------------------------------------------------------------
    def _generate_test_case(
        self,
        module_name: str,
        name: str,
        info: Dict[str, Any],
        props: List[str],
    ) -> List[str]:

        test_name = f"test_property_{name.replace('.', '_')}"
        lines: List[str] = [f"def {test_name}():"]

        # ------------------------------------------------------------
        # Case 1: free function
        # ------------------------------------------------------------
        if "." not in name:
            func_name = name

            # 🔥 Source‑Modul korrekt referenzieren
            lines.append(indent(f"func = {module_name}.{func_name}", 4))
            lines.append(indent("assert callable(func)", 4))

            dummy_args = self._build_dummy_args(info.get("args", {}))

            for prop in props:
                if prop == "no_crash_on_valid_input":
                    lines.extend(self._prop_no_crash(dummy_args))
                elif prop == "deterministic_output":
                    lines.extend(self._prop_deterministic(dummy_args))
                elif prop == "output_type_consistent":
                    lines.extend(self._prop_output_type(info, dummy_args))

            return lines

        # ------------------------------------------------------------
        # Case 2: class method (no instantiation, no calls)
        # ------------------------------------------------------------
        cls_name, method_name = name.split(".")

        # 🔥 Source‑Modul korrekt referenzieren
        lines.append(indent(f"cls = {module_name}.{cls_name}", 4))
        lines.append(indent("assert callable(cls)", 4))
        lines.append(indent(f"assert hasattr(cls, '{method_name}')", 4))
        lines.append(indent("# property tests for methods: existence and callability only; instantiation skipped", 4))

        return lines

    # ------------------------------------------------------------
    # Property: no crash on valid input
    # ------------------------------------------------------------
    def _prop_no_crash(self, dummy_args: Optional[str]) -> List[str]:
        lines = [indent("# property: no crash on valid input", 4)]

        if dummy_args is None:
            lines.append(indent("assert True  # no args to test", 4))
            return lines

        if self.enable_runtime_checks:
            call = "func()" if dummy_args == "" else f"func({dummy_args})"
            lines.append(indent(call, 4))
            lines.append(indent("assert True", 4))
        else:
            lines.append(indent("assert True  # runtime checks disabled", 4))

        return lines

    # ------------------------------------------------------------
    # Property: deterministic output
    # ------------------------------------------------------------
    def _prop_deterministic(self, dummy_args: Optional[str]) -> List[str]:
        lines = [indent("# property: deterministic output", 4)]

        if dummy_args is None:
            lines.append(indent("assert True  # no args to test", 4))
            return lines

        if self.enable_runtime_checks:
            if dummy_args == "":
                lines.append(indent("r1 = func()", 4))
                lines.append(indent("r2 = func()", 4))
            else:
                lines.append(indent(f"r1 = func({dummy_args})", 4))
                lines.append(indent(f"r2 = func({dummy_args})", 4))
            lines.append(indent("assert r1 == r2", 4))
        else:
            lines.append(indent("assert True  # runtime checks disabled", 4))

        return lines

    # ------------------------------------------------------------
    # Property: output type consistent
    # ------------------------------------------------------------
    def _prop_output_type(self, info: Dict[str, Any], dummy_args: Optional[str]) -> List[str]:
        lines = [indent("# property: output type consistent", 4)]
        expected = info.get("return")

        if dummy_args is None or not expected:
            lines.append(indent("assert True  # insufficient type info", 4))
            return lines

        if self.enable_runtime_checks:
            call = "func()" if dummy_args == "" else f"func({dummy_args})"
            lines.append(indent(f"result = {call}", 4))
            py_type = self._expected_runtime_type(expected)
            if py_type:
                lines.append(indent(f"assert isinstance(result, {py_type})", 4))
            else:
                lines.append(indent("assert True  # unknown type", 4))
        else:
            lines.append(indent("assert True  # runtime checks disabled", 4))

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
            return "1"
        if lowered == "bool":
            return "True"
        if lowered == "str":
            return "'x'"
        if "list" in lowered or "tuple" in lowered or "set" in lowered:
            return "[1]"
        if "dict" in lowered or "mapping" in lowered:
            return "{'k': 1}"
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
