"""
SchemaBuilder

Corrected to:
- include constructor args
- include method args
- include function args
- include method dictionaries
- propagate ctor_args from classes to methods
- align with corrected ASTInspector + StaticAnalyzer + SemanticAnalyzer + TypeFusion
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


class SchemaBuilder:
    """
    Build a canonical schema from fused inference results.
    """

    def __init__(self, settings: Dict[str, Any], structure_registry) -> None:
        self.settings = settings
        self.registry = structure_registry

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def build(
        self,
        file_path: Path,
        fused_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        schema: Dict[str, Any] = {}

        # First pass: build entries
        for name, info in fused_info.items():
            kind = info.get("kind")

            entry: Dict[str, Any] = {
                "kind": kind,
                "return": info.get("return"),
                "intent": info.get("intent"),
                "behavior": info.get("behavior", {}),
                "confidence": info.get("confidence", 0.0),
                "docstring": info.get("docstring"),
            }

            # ------------------------------------------------------------
            # Correct handling of constructor vs method/function args
            # ------------------------------------------------------------
            if kind == "class":
                # Constructor arguments
                entry["ctor_args"] = info.get("args", {})

                # Methods dictionary (from ASTInspector → StaticAnalyzer → TypeFusion)
                entry["methods"] = info.get("methods", {})

            else:
                # Functions and methods use args normally
                entry["args"] = info.get("args", {})

            schema[name] = entry

        # ------------------------------------------------------------
        # Second pass: propagate ctor_args from class to methods
        # ------------------------------------------------------------
        for name, entry in schema.items():
            if entry.get("kind") == "method" and "." in name:
                cls_name, _ = name.split(".", 1)
                cls_entry = schema.get(cls_name, {})
                ctor_args = cls_entry.get("ctor_args", {})
                # Attach ctor_args to method entry so generators can see them
                entry["ctor_args"] = ctor_args

        # Store schema in registry
        self.registry.store_schema(file_path, schema)

        return schema

    # ------------------------------------------------------------
    # Retrieve stored schema
    # ------------------------------------------------------------
    def get_schema(self, file_path: Path) -> Dict[str, Any]:
        return self.registry.get_schema(file_path)

    # ------------------------------------------------------------
    # Summary helper
    # ------------------------------------------------------------
    def summarize(self, schema: Dict[str, Any]) -> str:
        lines = ["=== Schema Summary ===", ""]
        for name, info in schema.items():
            lines.append(f"{name}:")
            lines.append(f"  kind: {info.get('kind')}")

            if info.get("kind") == "class":
                lines.append(f"  ctor_args: {info.get('ctor_args')}")
                lines.append(f"  methods: {list(info.get('methods', {}).keys())}")
            else:
                lines.append(f"  args: {info.get('args')}")
                lines.append(f"  ctor_args (propagated): {info.get('ctor_args')}")

            lines.append(f"  return: {info.get('return')}")
            lines.append(f"  intent: {info.get('intent')}")
            lines.append(f"  behavior: {info.get('behavior')}")
            lines.append(f"  confidence: {info.get('confidence')}")
            lines.append("")
        return "\n".join(lines)
