"""
DynamicProbe

Corrected to:
- include dynamic argument types
- include dynamic return types
- avoid unsafe execution
- align with corrected ASTInspector + TypeFusion
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Any, Optional

from core.safe_import import SafeImporter


class DynamicProbe:
    """
    Perform safe dynamic probing on Python modules.
    """

    def __init__(self, settings: Dict[str, Any], safe_importer: SafeImporter) -> None:
        self.settings = settings
        self.safe_importer = safe_importer

        # Only allow probing functions with zero arguments
        self.allow_zero_arg_calls: bool = settings["inference"]["dynamic_probe"]["allow_zero_arg_calls"]

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def probe(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        file_path = structure.get("__file__")
        if not file_path:
            return results

        module = self.safe_importer.import_file(Path(file_path))
        if module is None:
            return results

        classes = structure.get("classes", {})
        functions = structure.get("functions", {})

        # ------------------------------------------------------------
        # Probe classes
        # ------------------------------------------------------------
        for cls_name, cls_info in classes.items():
            cls_obj = getattr(module, cls_name, None)
            if cls_obj is None:
                continue

            for method_name in cls_info.get("methods", {}):
                full_name = f"{cls_name}.{method_name}"
                method_obj = getattr(cls_obj, method_name, None)
                results[full_name] = self._probe_callable(method_obj)

        # ------------------------------------------------------------
        # Probe top-level functions
        # ------------------------------------------------------------
        for func_name in functions.keys():
            func_obj = getattr(module, func_name, None)
            results[func_name] = self._probe_callable(func_obj)

        return results

    # ------------------------------------------------------------
    # Callable probing
    # ------------------------------------------------------------
    def _probe_callable(self, obj: Any) -> Dict[str, Any]:
        info = {
            "callable": False,
            "arity": 0,
            "defaults": {},
            "arg_types": {},
            "safe_return_type": None,
        }

        if not callable(obj):
            return info

        info["callable"] = True

        try:
            sig = inspect.signature(obj)
        except Exception:
            return info

        params = list(sig.parameters.values())
        info["arity"] = len(params)

        # Default values + dynamic arg types
        defaults = {}
        arg_types = {}

        for p in params:
            if p.default is not inspect._empty:
                defaults[p.name] = repr(p.default)

            # Infer simple semantic type from annotation
            if p.annotation is not inspect._empty:
                try:
                    ann = str(p.annotation)
                except Exception:
                    ann = None
                arg_types[p.name] = self._infer_annotation_type(ann)
            else:
                arg_types[p.name] = None

        info["defaults"] = defaults
        info["arg_types"] = arg_types

        # Safe zero-arg probing
        if self.allow_zero_arg_calls and info["arity"] == 0:
            try:
                result = obj()
                info["safe_return_type"] = self._infer_return_type(result)
            except Exception:
                info["safe_return_type"] = None

        return info

    # ------------------------------------------------------------
    # Annotation type inference
    # ------------------------------------------------------------
    def _infer_annotation_type(self, ann: Optional[str]) -> Optional[str]:
        if not ann:
            return None

        lowered = ann.lower()

        if lowered in ("int", "float", "complex"):
            return "numeric"
        if lowered == "bool":
            return "boolean"
        if lowered == "str":
            return "text"
        if any(x in lowered for x in ("list", "tuple", "set")):
            return "collection"
        if "dict" in lowered or "mapping" in lowered:
            return "mapping"

        return "unknown"

    # ------------------------------------------------------------
    # Return type inference
    # ------------------------------------------------------------
    def _infer_return_type(self, value: Any) -> Optional[str]:
        if value is None:
            return "none"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, (int, float, complex)):
            return "numeric"
        if isinstance(value, str):
            return "text"
        if isinstance(value, (list, tuple, set)):
            return "collection"
        if isinstance(value, dict):
            return "mapping"
        return "unknown"
