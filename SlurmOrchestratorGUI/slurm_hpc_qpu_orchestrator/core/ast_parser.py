"""
ASTParser
---------
Safely parses an uploaded Python workflow file and extracts structural
information needed for workflow classification (classical, quantum, hybrid).

This module NEVER executes user code. All analysis is static.
"""

import ast
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structure returned by ASTParser
# ---------------------------------------------------------------------------

@dataclass
class ParsedWorkflow:
    imports: List[str]
    function_calls: List[str]
    has_loops: bool
    file_path: Optional[Path]


# ---------------------------------------------------------------------------
# AST Parser
# ---------------------------------------------------------------------------

class ASTParser:
    """
    Safely parses Python files using the built-in AST module.
    Extracts:
        - import statements
        - function calls
        - presence of loops (for hybrid detection)
    """

    def __init__(self):
        pass

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def parse_file(self, file_path: Path) -> ParsedWorkflow:
        """
        Parse a Python file and extract structural information.

        Parameters
        ----------
        file_path : Path
            Path to the uploaded Python workflow file.

        Returns
        -------
        ParsedWorkflow
            Structured information extracted from the AST.
        """
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        imports = self._extract_imports(tree)
        calls = self._extract_function_calls(tree)
        has_loops = self._detect_loops(tree)

        return ParsedWorkflow(
            imports=imports,
            function_calls=calls,
            has_loops=has_loops,
            file_path=file_path
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """
        Extract all import statements from the AST.
        """
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return imports

    def _extract_function_calls(self, tree: ast.AST) -> List[str]:
        """
        Extract all function call names from the AST.
        """
        calls = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name:
                    calls.append(func_name)

        return calls

    def _detect_loops(self, tree: ast.AST) -> bool:
        """
        Detect presence of loops (for hybrid workflow detection).
        """
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                return True
        return False

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """
        Extract the function name from a Call node.
        Handles:
            - direct calls: func()
            - attribute calls: obj.func()
        """
        if isinstance(node.func, ast.Name):
            return node.func.id

        if isinstance(node.func, ast.Attribute):
            return f"{self._get_attr_chain(node.func)}"

        return None

    def _get_attr_chain(self, attr: ast.Attribute) -> str:
        """
        Build dotted attribute chain (e.g., qiskit_ibm_runtime.Sampler.run)
        """
        parts = []
        current = attr

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            parts.append(current.id)

        return ".".join(reversed(parts))