"""
ASTInspector

Extended to extract:
- class constructor (__init__) parameters
- method parameters
- function parameters
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Any, List, Optional


class ASTInspector:
    """
    Parse Python source files and extract structural information.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings: Dict[str, Any] = settings

        self.include_private: bool = settings["core"]["ast_inspector"]["include_private_functions"]
        self.include_magic: bool = settings["core"]["ast_inspector"]["include_magic_methods"]

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def inspect_file(self, file_path: Path) -> Dict[str, Any]:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        classes = self._extract_classes(tree)
        functions = self._extract_functions(tree)
        docstrings = self._extract_docstrings(tree)
        annotations = self._extract_annotations(tree)

        return {
            "classes": classes,
            "functions": functions,
            "docstrings": docstrings,
            "annotations": annotations,
        }

    # ------------------------------------------------------------
    # Class extraction (now includes constructor + method args)
    # ------------------------------------------------------------
    def _extract_classes(self, tree: ast.AST) -> Dict[str, Dict[str, Any]]:
        classes: Dict[str, Dict[str, Any]] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name

                ctor_args: Dict[str, Optional[str]] = {}
                methods: Dict[str, Dict[str, Optional[str]]] = {}

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if not self._should_include(item.name):
                            continue

                        # Constructor
                        if item.name == "__init__":
                            ctor_args = self._extract_parameters(item)

                        # Methods
                        else:
                            methods[item.name] = self._extract_parameters(item)

                classes[class_name] = {
                    "ctor_args": ctor_args,
                    "methods": methods,
                }

        return classes

    # ------------------------------------------------------------
    # Function extraction (now includes parameters)
    # ------------------------------------------------------------
    def _extract_functions(self, tree: ast.AST) -> Dict[str, Dict[str, Optional[str]]]:
        functions: Dict[str, Dict[str, Optional[str]]] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if isinstance(node.parent, ast.ClassDef):
                    continue

                if not self._should_include(node.name):
                    continue

                functions[node.name] = self._extract_parameters(node)

        return functions

    # ------------------------------------------------------------
    # Parameter extraction helper
    # ------------------------------------------------------------
    def _extract_parameters(self, func_node: ast.FunctionDef) -> Dict[str, Optional[str]]:
        params: Dict[str, Optional[str]] = {}

        # Skip "self"
        args = func_node.args.args[1:] if func_node.args.args else []

        for arg in args:
            ann = None
            if arg.annotation:
                try:
                    ann = ast.unparse(arg.annotation)
                except Exception:
                    ann = "<unknown>"
            params[arg.arg] = ann

        return params

    # ------------------------------------------------------------
    # Docstring extraction
    # ------------------------------------------------------------
    def _extract_docstrings(self, tree: ast.AST) -> Dict[str, str]:
        docstrings: Dict[str, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                if doc:
                    docstrings[node.name] = doc

        return docstrings

    # ------------------------------------------------------------
    # Return annotation extraction
    # ------------------------------------------------------------
    def _extract_annotations(self, tree: ast.AST) -> Dict[str, str]:
        annotations: Dict[str, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns:
                    try:
                        annotations[node.name] = ast.unparse(node.returns)
                    except Exception:
                        annotations[node.name] = "<unknown>"

        return annotations

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _should_include(self, name: str) -> bool:
        if name.startswith("_") and not self.include_private:
            return False

        if name.startswith("__") and name.endswith("__") and not self.include_magic:
            return False

        return True


# ------------------------------------------------------------
# Parent attachment
# ------------------------------------------------------------
def _attach_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node


_original_parse = ast.parse


def parse_with_parents(source: str) -> ast.AST:
    tree = _original_parse(source)
    _attach_parents(tree)
    return tree


ast.parse = parse_with_parents
