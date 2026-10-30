"""
script_manager.py

This module is responsible for:
- Validating uploaded Python scripts
- Extracting metadata (hash, size, imports, entry points)
- Providing a stable API for other backend modules
- Managing analysis cache (to avoid re-parsing unchanged scripts)
"""

from pathlib import Path
import hashlib
import ast
import json
from dataclasses import dataclass
from typing import Optional, List, Dict


# ============================================================
# Data structures
# ============================================================

@dataclass
class ScriptMetadata:
    """Metadata describing a Python script."""
    path: Path
    hash: str
    size_bytes: int
    imports: List[str]
    entry_points: List[str]


@dataclass
class ScriptLoadResult:
    """Result returned when loading a script."""
    metadata: ScriptMetadata
    ast_tree: ast.AST
    cached: bool


# ============================================================
# Script Manager
# ============================================================

class ScriptManager:
    """
    Handles loading, validating, hashing, and caching of Python scripts.
    This is the first module used by the GUI and backend pipeline.
    """

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache_path.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.cache_path / "analysis_cache.json"

        # Load cache if exists
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except json.JSONDecodeError:
                self.cache = {}
        else:
            self.cache = {}

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def load_script(self, script_path: Path) -> ScriptLoadResult:
        """
        Main entry point for loading and validating a Python script.
        Returns metadata, AST tree, and cache status.
        """
        self._validate_path(script_path)

        script_hash = self._compute_hash(script_path)
        metadata = self._extract_metadata(script_path, script_hash)

        cached = script_hash in self.cache

        # Parse AST
        try:
            ast_tree = ast.parse(script_path.read_text())
        except SyntaxError as e:
            raise SyntaxError(f"Failed to parse script '{script_path}': {e}")

        # Update cache if needed
        if not cached:
            self.cache[script_hash] = {
                "imports": metadata.imports,
                "entry_points": metadata.entry_points,
                "size_bytes": metadata.size_bytes,
            }
            self._write_cache()

        return ScriptLoadResult(
            metadata=metadata,
            ast_tree=ast_tree,
            cached=cached
        )

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _validate_path(self, path: Path):
        """Ensure the file exists and is a Python script."""
        if not path.exists():
            raise FileNotFoundError(f"Script not found: {path}")
        if path.suffix.lower() != ".py":
            raise ValueError("Uploaded file must be a .py script")

    def _compute_hash(self, path: Path) -> str:
        """Compute SHA256 hash of the script contents."""
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def _extract_metadata(self, path: Path, script_hash: str) -> ScriptMetadata:
        """Extract imports, entry points, and size."""
        text = path.read_text()
        tree = ast.parse(text)

        imports = self._extract_imports(tree)
        entry_points = self._extract_entry_points(tree)

        return ScriptMetadata(
            path=path,
            hash=script_hash,
            size_bytes=path.stat().st_size,
            imports=imports,
            entry_points=entry_points
        )

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imported modules."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    def _extract_entry_points(self, tree: ast.AST) -> List[str]:
        """
        Detect functions that could serve as entry points.
        Convention: main(), run(), execute()
        """
        entry_points = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if node.name in ("main", "run", "execute"):
                    entry_points.append(node.name)
        return entry_points

    def _write_cache(self):
        """Write cache to disk."""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)
