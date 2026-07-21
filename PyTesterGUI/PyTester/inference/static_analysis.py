"""
StaticAnalyzer

Corrected to:
- include constructor args from ASTInspector
- include method args from ASTInspector
- include function args from ASTInspector
- merge annotation extractor results properly
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from core.annotation_extractor import AnnotationExtractor
from core.docstring_extractor import DocstringExtractor
from core.utils import safe_repr


class StaticAnalyzer:
    """
    Perform static analysis on Python source files and their structures.
    """

    def __init__(
        self,
        settings: Dict[str, Any],
        annotation_extractor: AnnotationExtractor,
        docstring_extractor: DocstringExtractor,
    ) -> None:
        self.settings = settings
        self.annotation_extractor = annotation_extractor
        self.docstring_extractor = docstring_extractor

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def analyze(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform static analysis on a structure dictionary.
        """
        results: Dict[str, Any] = {}

        classes = structure.get("classes", {})
        functions = structure.get("functions", {})
        docstrings = structure.get("docstrings", {})
        annotations = structure.get("annotations", {})

        # ------------------------------------------------------------
        # Classes + methods
        # ------------------------------------------------------------
        for cls_name, cls_info in classes.items():
            ctor_args = cls_info.get("ctor_args", {})
            methods = cls_info.get("methods", {})

            # Class entry
            results[cls_name] = {
                "kind": "class",
                "args": ctor_args,                     # CORRECT: constructor args
                "return": None,
                "docstring": docstrings.get(cls_name),
                "properties": {
                    "has_methods": bool(methods),
                },
            }

            # Method entries
            for method_name, method_args in methods.items():
                full_name = f"{cls_name}.{method_name}"

                return_ann = annotations.get(method_name)

                results[full_name] = {
                    "kind": "method",
                    "args": method_args,                # CORRECT: method args
                    "return": return_ann,
                    "docstring": docstrings.get(method_name),
                    "properties": {
                        "belongs_to": cls_name,
                    },
                }

        # ------------------------------------------------------------
        # Top-level functions
        # ------------------------------------------------------------
        for func_name, func_args in functions.items():
            return_ann = annotations.get(func_name)

            results[func_name] = {
                "kind": "function",
                "args": func_args,                     # CORRECT: function args
                "return": return_ann,
                "docstring": docstrings.get(func_name),
                "properties": {},
            }

        return results

    # ------------------------------------------------------------
    # Optional file-based analysis
    # ------------------------------------------------------------
    def analyze_file(self, file_path: Path, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform static analysis using both structure and raw file annotations.
        """
        base = self.analyze(structure)
        file_annotations = self.annotation_extractor.extract(file_path)
        file_docstrings = self.docstring_extractor.extract(file_path)

        for name, info in base.items():
            # Merge docstrings from file-level extractor if missing
            if info.get("docstring") is None and name in file_docstrings:
                info["docstring"] = file_docstrings[name]

            # Merge argument annotations if available
            if name in file_annotations:
                ann = file_annotations[name]

                # Merge args
                if ann.get("args"):
                    info["args"] = ann.get("args")

                # Merge return annotation
                if info.get("return") is None and ann.get("return"):
                    info["return"] = ann.get("return")

        return base

    # ------------------------------------------------------------
    # Summary helper
    # ------------------------------------------------------------
    def summarize(self, static_info: Dict[str, Any]) -> str:
        """
        Produce a human-readable summary of static analysis results.
        """
        lines = ["=== Static Analysis Summary ===", ""]
        for name, info in static_info.items():
            lines.append(f"{name}:")
            lines.append(f"  kind: {info.get('kind')}")
            lines.append(f"  args: {safe_repr(info.get('args'))}")
            lines.append(f"  return: {safe_repr(info.get('return'))}")
            lines.append(f"  docstring: {safe_repr(info.get('docstring'))}")
            lines.append(f"  properties: {safe_repr(info.get('properties'))}")
            lines.append("")
        return "\n".join(lines)
