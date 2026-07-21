"""
SemanticAnalyzer

Corrected to:
- include constructor args from ASTInspector
- include method args from ASTInspector
- include function args from ASTInspector
- infer semantic types for all arguments
"""

from __future__ import annotations

from typing import Dict, Any, Optional


class SemanticAnalyzer:
    """
    Perform semantic analysis on Python structures.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def analyze(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        classes = structure.get("classes", {})
        functions = structure.get("functions", {})
        docstrings = structure.get("docstrings", {})
        annotations = structure.get("annotations", {})

        # ------------------------------------------------------------
        # Classes
        # ------------------------------------------------------------
        for cls_name, cls_info in classes.items():
            ctor_args = cls_info.get("ctor_args", {})
            methods = cls_info.get("methods", {})

            results[cls_name] = {
                "intent": self._infer_intent(docstrings.get(cls_name)),
                "arg_semantics": self._infer_arg_semantics(ctor_args),
                "return_semantics": None,
                "behavior": {
                    "is_class": True,
                    "has_methods": bool(methods),
                },
            }

            # Methods
            for method_name, method_args in methods.items():
                full_name = f"{cls_name}.{method_name}"
                return_ann = annotations.get(method_name)

                results[full_name] = {
                    "intent": self._infer_intent(docstrings.get(method_name)),
                    "arg_semantics": self._infer_arg_semantics(method_args),
                    "return_semantics": self._infer_semantic_type(return_ann),
                    "behavior": self._infer_behavior(method_name, docstrings.get(method_name)),
                }

        # ------------------------------------------------------------
        # Top-level functions
        # ------------------------------------------------------------
        for func_name, func_args in functions.items():
            return_ann = annotations.get(func_name)

            results[func_name] = {
                "intent": self._infer_intent(docstrings.get(func_name)),
                "arg_semantics": self._infer_arg_semantics(func_args),
                "return_semantics": self._infer_semantic_type(return_ann),
                "behavior": self._infer_behavior(func_name, docstrings.get(func_name)),
            }

        return results

    # ------------------------------------------------------------
    # Intent inference
    # ------------------------------------------------------------
    def _infer_intent(self, doc: Optional[str]) -> Optional[str]:
        if not doc:
            return None

        lowered = doc.lower()

        if "validate" in lowered or "check" in lowered:
            return "validation"
        if "compute" in lowered or "calculate" in lowered:
            return "computation"
        if "transform" in lowered or "convert" in lowered:
            return "transformation"
        if "load" in lowered or "read" in lowered:
            return "io-read"
        if "write" in lowered or "save" in lowered:
            return "io-write"

        return None

    # ------------------------------------------------------------
    # Argument semantic type inference
    # ------------------------------------------------------------
    def _infer_arg_semantics(self, arg_dict: Dict[str, Optional[str]]) -> Dict[str, str]:
        semantics: Dict[str, str] = {}

        for arg_name, ann in arg_dict.items():
            semantics[arg_name] = self._infer_semantic_type(ann)

        return semantics

    # ------------------------------------------------------------
    # Semantic type inference
    # ------------------------------------------------------------
    def _infer_semantic_type(self, annotation: Optional[str]) -> Optional[str]:
        if not annotation:
            return None

        ann = annotation.lower()

        # Optional[T]
        if "optional" in ann:
            inner = ann.replace("optional[", "").replace("]", "")
            return self._infer_semantic_type(inner)

        # Numeric
        if ann in ("int", "float", "complex"):
            return "numeric"

        # Boolean
        if ann == "bool":
            return "boolean"

        # Text
        if ann == "str":
            return "text"

        # Collections
        if any(x in ann for x in ("list", "tuple", "set")):
            return "collection"

        # Mapping
        if "dict" in ann or "mapping" in ann:
            return "mapping"

        # Callable
        if "callable" in ann:
            return "callable"

        return "unknown"

    # ------------------------------------------------------------
    # Behavior inference
    # ------------------------------------------------------------
    def _infer_behavior(self, name: str, doc: Optional[str]) -> Dict[str, Any]:
        behavior: Dict[str, Any] = {}

        lowered_name = name.lower()
        lowered_doc = doc.lower() if doc else ""

        # IO behavior
        if any(x in lowered_name for x in ("load", "read", "fetch")):
            behavior["io"] = "read"
        if any(x in lowered_name for x in ("save", "write", "store")):
            behavior["io"] = "write"

        # Mutation behavior
        if any(x in lowered_name for x in ("update", "modify", "set")):
            behavior["mutates_state"] = True

        # Pure computation
        if any(x in lowered_name for x in ("compute", "calculate", "eval")):
            behavior["pure"] = True

        # Error behavior
        if "raise" in lowered_doc or "error" in lowered_doc:
            behavior["may_raise"] = True

        return behavior

    # ------------------------------------------------------------
    # Summary helper
    # ------------------------------------------------------------
    def summarize(self, semantic_info: Dict[str, Any]) -> str:
        lines = ["=== Semantic Analysis Summary ===", ""]
        for name, info in semantic_info.items():
            lines.append(f"{name}:")
            lines.append(f"  intent: {info.get('intent')}")
            lines.append(f"  arg_semantics: {info.get('arg_semantics')}")
            lines.append(f"  return_semantics: {info.get('return_semantics')}")
            lines.append(f"  behavior: {info.get('behavior')}")
            lines.append("")
        return "\n".join(lines)
