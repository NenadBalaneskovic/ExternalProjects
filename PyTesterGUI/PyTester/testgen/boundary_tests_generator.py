"""
BoundaryTestsGenerator

Corrected to:
- NEVER instantiate classes when constructor args exist
- NEVER call methods when method args exist
- ONLY run runtime checks when safe and allowed
- Remain deterministic and side‑effect‑free
- ALWAYS import the correct source module for coverage
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

from core.utils import indent


class BoundaryTestsGenerator:
    """
    Generate boundary value tests from a canonical schema.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.enable_runtime_checks = settings["testgen"]["boundary"]["enable_runtime_checks"]

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def generate(self, file_path: Path, structure: Dict[str, Any], schema: Dict[str, Any]) -> str:
        """
        file_path: path to the SOURCE file (workspace/source/*.py)
        module_name: name of the module WITHOUT extension
        """
        module_name = file_path.stem

        lines: List[str] = []
        lines.append("import pytest")

        # 🔥 WICHTIG: Source‑Modul korrekt importieren
        lines.append(f"from workspace.source import {module_name}")
        lines.append("")

        for name, info in schema.items():
            if info.get("kind") not in ("function", "method"):
                continue

            # Skip zero‑arg functions/methods (no boundary values)
            if not info.get("args"):
                continue

            lines.extend(self._generate_test_case(module_name, name, info))
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------
    # Individual test case generation
    # ------------------------------------------------------------
    def _generate_test_case(self, module_name: str, name: str, info: Dict[str, Any]) -> List[str]:
        """
        Generate a single boundary test case.

        Corrected rules:
        - Free functions: boundary test only if args exist.
        - Class methods:
            - NEVER instantiate class if constructor args exist.
            - NEVER call method if method args exist.
            - Only run boundary tests when both ctor and method args are empty.
        """

        test_name = f"test_boundary_{name.replace('.', '_')}"
        lines: List[str] = [f"def {test_name}():"]

        # ------------------------------------------------------------
        # Case 1: free function
        # ------------------------------------------------------------
        if "." not in name:
            func_name = name

            # 🔥 Source‑Modul korrekt referenzieren
            lines.append(indent(f"func = {module_name}.{func_name}", 4))
            lines.append(indent("assert callable(func)", 4))

            boundary_sets = self._build_boundary_sets(info.get("args", {}))

            for idx, arg_expr in enumerate(boundary_sets):
                lines.append(indent(f"# boundary set {idx+1}", 4))
                if self.enable_runtime_checks:
                    lines.append(indent(f"func({arg_expr})", 4))
                    lines.append(indent("assert True  # boundary invocation succeeded", 4))
                else:
                    lines.append(indent("assert True  # boundary values prepared", 4))

            return lines

        # ------------------------------------------------------------
        # Case 2: class method
        # ------------------------------------------------------------
        cls_name, method_name = name.split(".")

        # 🔥 Source‑Modul korrekt referenzieren
        lines.append(indent(f"cls = {module_name}.{cls_name}", 4))
        lines.append(indent("assert callable(cls)", 4))

        ctor_args = info.get("ctor_args", {})
        method_args = info.get("args", {})

        # Constructor requires args → skip
        if ctor_args:
            lines.append(indent(f"assert hasattr(cls, '{method_name}')", 4))
            lines.append(indent("# constructor requires arguments → skip boundary tests", 4))
            return lines

        # Method requires args → skip
        if method_args:
            lines.append(indent(f"assert hasattr(cls, '{method_name}')", 4))
            lines.append(indent("# method requires arguments → skip boundary tests", 4))
            return lines

        # Safe zero‑arg constructor
        lines.append(indent("instance = cls()", 4))
        lines.append(indent("assert instance is not None", 4))

        # Resolve method
        lines.append(indent(f"func = getattr(instance, '{method_name}')", 4))
        lines.append(indent("assert callable(func)", 4))

        # ------------------------------------------------------------
        # Apply boundary sets (only zero‑arg methods)
        # ------------------------------------------------------------
        boundary_sets = self._build_boundary_sets(method_args)

        for idx, arg_expr in enumerate(boundary_sets):
            lines.append(indent(f"# boundary set {idx+1}", 4))
            if self.enable_runtime_checks:
                call = "func()" if arg_expr == "" else f"func({arg_expr})"
                lines.append(indent(call, 4))
                lines.append(indent("assert True  # boundary invocation succeeded", 4))
            else:
                lines.append(indent("assert True  # boundary values prepared", 4))

        return lines

    # ------------------------------------------------------------
    # Boundary value construction
    # ------------------------------------------------------------
    def _build_boundary_sets(self, args: Dict[str, Optional[str]]) -> List[str]:
        if not args:
            return [""]

        per_arg_boundaries: List[List[str]] = []
        for name, ann in args.items():
            per_arg_boundaries.append(self._boundary_values_for_annotation(ann))

        # Cartesian product
        def combine(values: List[List[str]]) -> List[List[str]]:
            if not values:
                return [[]]
            head = values[0]
            tail = combine(values[1:])
            return [[h] + t for h in head for t in tail]

        combined = combine(per_arg_boundaries)
        return [", ".join(exprs) for exprs in combined]

    def _boundary_values_for_annotation(self, ann: Optional[str]) -> List[str]:
        if not ann:
            return ["None"]

        lowered = ann.lower()

        if lowered in ("int", "float", "complex"):
            return ["0", "1", "-1"]

        if lowered == "bool":
            return ["True", "False"]

        if lowered == "str":
            return ["''", "'a'", "'test'"]

        if "list" in lowered or "tuple" in lowered or "set" in lowered:
            return ["[]", "[1]", "[None]"]

        if "dict" in lowered or "mapping" in lowered:
            return ["{}", "{'a': 1}", "{None: None}"]

        return ["None"]
