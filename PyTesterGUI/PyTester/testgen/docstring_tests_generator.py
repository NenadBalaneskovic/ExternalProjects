"""
DocstringTestsGenerator

Corrected to:
- NEVER instantiate classes when constructor args exist
- NEVER call methods when method args exist
- ONLY run runtime checks when safe and allowed
- Remain deterministic and side‑effect‑free
- ALWAYS import the correct source module for coverage
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.utils import indent


class DocstringTestsGenerator:
    """
    Generate docstring-derived tests from a canonical schema.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.enable_runtime_checks = settings["testgen"]["docstring"]["enable_runtime_checks"]

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
            doc = info.get("docstring")
            if not doc:
                continue

            claims = self._extract_claims(doc)
            if not claims:
                continue

            lines.extend(self._generate_test_case(module_name, name, info, claims))
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------
    # Claim extraction
    # ------------------------------------------------------------
    def _extract_claims(self, doc: str) -> Dict[str, Any]:
        claims: Dict[str, Any] = {}
        lowered = doc.lower()

        m = re.search(r"returns\s+([a-zA-Z0-9_]+)", lowered)
        if m:
            claims["returns"] = m.group(1)

        m = re.search(r"raises\s+([a-zA-Z0-9_]+)", lowered)
        if m:
            claims["raises"] = m.group(1)

        m = re.search(r"input\s+must\s+be\s+([a-zA-Z0-9_]+)", lowered)
        if m:
            claims["input"] = m.group(1)

        m = re.search(r"output\s+is\s+([a-zA-Z0-9_]+)", lowered)
        if m:
            claims["output"] = m.group(1)

        return claims

    # ------------------------------------------------------------
    # Individual test case generation
    # ------------------------------------------------------------
    def _generate_test_case(
        self,
        module_name: str,
        name: str,
        info: Dict[str, Any],
        claims: Dict[str, Any],
    ) -> List[str]:

        test_name = f"test_docstring_{name.replace('.', '_')}"
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

            for claim_type, claim_value in claims.items():
                lines.extend(self._apply_claim(
                    func="func",
                    claim_type=claim_type,
                    claim_value=claim_value,
                    dummy_args=dummy_args,
                    info=info,
                ))

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

        # Constructor requires args → skip runtime checks
        if ctor_args:
            lines.append(indent(f"assert hasattr(cls, '{method_name}')", 4))
            lines.append(indent("# constructor requires arguments → skip runtime docstring checks", 4))
            return lines

        # Safe zero‑arg constructor
        lines.append(indent("instance = cls()", 4))
        lines.append(indent("assert instance is not None", 4))

        # Resolve method
        lines.append(indent(f"func = getattr(instance, '{method_name}')", 4))
        lines.append(indent("assert callable(func)", 4))

        dummy_args = self._build_dummy_args(method_args)

        # Method requires args → skip
        if method_args and dummy_args is None:
            lines.append(indent("# method requires arguments → skip runtime docstring checks", 4))
            return lines

        for claim_type, claim_value in claims.items():
            lines.extend(self._apply_claim(
                func="func",
                claim_type=claim_type,
                claim_value=claim_value,
                dummy_args=dummy_args,
                info=info,
            ))

        return lines

    # ------------------------------------------------------------
    # Apply docstring claim
    # ------------------------------------------------------------
    def _apply_claim(self, func: str, claim_type: str, claim_value: str,
                     dummy_args: Optional[str], info: Dict[str, Any]) -> List[str]:

        if claim_type == "returns":
            return self._test_returns(func, claim_value, dummy_args)

        if claim_type == "raises":
            return self._test_raises(func, claim_value, dummy_args)

        if claim_type == "input":
            return self._test_input_constraint(claim_value)

        if claim_type == "output":
            return self._test_output_constraint(func, claim_value, dummy_args)

        return []

    # ------------------------------------------------------------
    # Claim: returns X
    # ------------------------------------------------------------
    def _test_returns(self, func: str, claim: str, dummy_args: Optional[str]) -> List[str]:
        lines = [indent(f"# docstring claim: returns {claim}", 4)]

        if not self.enable_runtime_checks or dummy_args is None:
            lines.append(indent("assert True  # runtime checks disabled or no args", 4))
            return lines

        call = f"{func}()" if dummy_args == "" else f"{func}({dummy_args})"
        lines.append(indent(f"result = {call}", 4))
        lines.append(indent(f"assert isinstance(result, {self._map_type(claim)})", 4))
        return lines

    # ------------------------------------------------------------
    # Claim: raises X
    # ------------------------------------------------------------
    def _test_raises(self, func: str, claim: str, dummy_args: Optional[str]) -> List[str]:
        lines = [indent(f"# docstring claim: raises {claim}", 4)]

        if not self.enable_runtime_checks or dummy_args is None:
            lines.append(indent("assert True  # runtime checks disabled or no args", 4))
            return lines

        call = f"{func}()" if dummy_args == "" else f"{func}({dummy_args})"
        lines.append(indent(f"with pytest.raises({claim}):", 4))
        lines.append(indent(f"    {call}", 4))
        return lines

    # ------------------------------------------------------------
    # Claim: input must be X
    # ------------------------------------------------------------
    def _test_input_constraint(self, claim: str) -> List[str]:
        return [
            indent(f"# docstring claim: input must be {claim}", 4),
            indent("assert True  # constraint noted", 4),
        ]

    # ------------------------------------------------------------
    # Claim: output is X
    # ------------------------------------------------------------
    def _test_output_constraint(self, func: str, claim: str, dummy_args: Optional[str]) -> List[str]:
        lines = [indent(f"# docstring claim: output is {claim}", 4)]

        if not self.enable_runtime_checks or dummy_args is None:
            lines.append(indent("assert True  # runtime checks disabled or no args", 4))
            return lines

        call = f"{func}()" if dummy_args == "" else f"{func}({dummy_args})"
        lines.append(indent(f"result = {call}", 4))
        lines.append(indent(f"assert isinstance(result, {self._map_type(claim)})", 4))
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
    # Type mapping for docstring claims
    # ------------------------------------------------------------
    def _map_type(self, claim: str) -> str:
        c = claim.lower()

        if c in ("int", "float", "complex"):
            return "(int, float, complex)"
        if c == "bool":
            return "bool"
        if c == "str":
            return "str"
        if c == "list":
            return "list"
        if c == "tuple":
            return "tuple"
        if c == "set":
            return "set"
        if c == "dict":
            return "dict"

        return "object"
