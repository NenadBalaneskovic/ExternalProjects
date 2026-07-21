"""
AnnotationExtractor

This subsystem is responsible for:
- extracting type annotations from Python source files
- supporting function arguments and return types
- producing a deterministic annotation dictionary

It is intentionally pure:
no execution, no imports, no dynamic behavior.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Any, Optional


class AnnotationExtractor:
    """
    Extract type annotations from Python source files.

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
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract all type annotations from a Python file.

        Parameters
        ----------
        file_path : Path
            Path to the Python file to inspect.

        Returns
        -------
        dict
            {
                function_name: {
                    "args": { arg_name: annotation_string },
                    "return": annotation_string or None
                }
            }
        """
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        annotations: Dict[str, Any] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name

                # Extract argument annotations
                arg_annotations: Dict[str, str] = {}
                for arg in node.args.args:
                    if arg.annotation:
                        arg_annotations[arg.arg] = self._annotation_to_str(arg.annotation)

                # Extract return annotation
                return_annotation: Optional[str] = None
                if node.returns:
                    return_annotation = self._annotation_to_str(node.returns)

                annotations[func_name] = {
                    "args": arg_annotations,
                    "return": return_annotation,
                }

        return annotations

    # ------------------------------------------------------------
    # Convenience helper
    # ------------------------------------------------------------
    def get_annotations(self, file_path: Path, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve annotations for a specific function.

        Parameters
        ----------
        file_path : Path
            Python file to inspect.

        name : str
            Name of the function.

        Returns
        -------
        Optional[dict]
            Annotation dictionary or None if not found.
        """
        all_annotations = self.extract(file_path)
        return all_annotations.get(name)

    # ------------------------------------------------------------
    # Annotation formatting
    # ------------------------------------------------------------
    def _annotation_to_str(self, node: ast.AST) -> str:
        """
        Convert an annotation AST node into a readable string.

        Returns
        -------
        str
            Human-readable annotation string.
        """
        try:
            return ast.unparse(node)
        except Exception:
            return "<unknown>"
