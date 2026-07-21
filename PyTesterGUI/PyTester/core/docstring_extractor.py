"""
DocstringExtractor

This subsystem is responsible for:
- extracting docstrings from Python source files
- supporting module-level, class-level, and function-level docstrings
- returning a clean, deterministic dictionary of docstrings

It is intentionally pure:
no execution, no imports, no dynamic behavior.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Any, Optional


class DocstringExtractor:
    """
    Extract docstrings from Python source files.

    Parameters
    ----------
    settings : dict
        Global configuration loaded from settings.yaml.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings: Dict[str, Any] = settings

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def extract(self, file_path: Path) -> Dict[str, str]:
        """
        Extract all docstrings from a Python file.

        Parameters
        ----------
        file_path : Path
            Path to the Python file to inspect.

        Returns
        -------
        dict
            { name: docstring }
        """
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        docstrings: Dict[str, str] = {}

        # Module-level docstring
        module_doc = ast.get_docstring(tree)
        if module_doc:
            docstrings["__module__"] = module_doc

        # Class + function docstrings
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                cls_doc = ast.get_docstring(node)
                if cls_doc:
                    docstrings[node.name] = cls_doc

            elif isinstance(node, ast.FunctionDef):
                func_doc = ast.get_docstring(node)
                if func_doc:
                    docstrings[node.name] = func_doc

        return docstrings

    # ------------------------------------------------------------
    # Convenience helper
    # ------------------------------------------------------------
    def get_docstring(self, file_path: Path, name: str) -> Optional[str]:
        """
        Retrieve a specific docstring by name.

        Parameters
        ----------
        file_path : Path
            Python file to inspect.

        name : str
            Name of the class/function/module.

        Returns
        -------
        Optional[str]
            Docstring text or None if not found.
        """
        all_docs = self.extract(file_path)
        return all_docs.get(name)
