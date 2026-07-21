"""
SyntaxChecker

This subsystem is responsible for:
- validating Python source files for syntax correctness
- providing detailed error messages for GUI display
- ensuring that only syntactically valid files enter the pipeline

It uses Python's built-in `ast` module to parse the file safely.
No execution, no imports, no dynamic behavior.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Any, Optional


class SyntaxChecker:
    """
    Validate Python source files for syntax correctness.

    Parameters
    ----------
    settings : dict
        Global configuration loaded from settings.yaml.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings: Dict[str, Any] = settings

    # ------------------------------------------------------------
    # Syntax checking
    # ------------------------------------------------------------
    def check_file(self, file_path: Path) -> bool:
        """
        Check whether a Python file contains valid syntax.

        Parameters
        ----------
        file_path : Path
            Path to the Python file to validate.

        Returns
        -------
        bool
            True if syntax is valid, False otherwise.
        """
        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception:
            return False

        try:
            ast.parse(source)
            return True
        except SyntaxError:
            return False

    # ------------------------------------------------------------
    # Detailed error reporting
    # ------------------------------------------------------------
    def get_syntax_errors(self, file_path: Path) -> Optional[str]:
        """
        Return a detailed syntax error message if the file is invalid.

        Parameters
        ----------
        file_path : Path
            Path to the Python file to validate.

        Returns
        -------
        Optional[str]
            Error message string, or None if syntax is valid.
        """
        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Could not read file: {exc}"

        try:
            ast.parse(source)
            return None
        except SyntaxError as err:
            return (
                f"SyntaxError: {err.msg}\n"
                f"Line: {err.lineno}, Offset: {err.offset}\n"
                f"Text: {err.text.strip() if err.text else ''}"
            )
