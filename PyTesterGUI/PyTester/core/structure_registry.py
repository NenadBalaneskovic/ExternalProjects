"""
StructureRegistry

This subsystem is responsible for:
- storing AST structures extracted by ASTInspector
- storing inference schemas produced by SchemaBuilder
- providing fast lookup for all GUI panels and backend subsystems

It acts as the central in-memory database for PyTester.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional


class StructureRegistry:
    """
    Registry for storing AST structures and inference schemas.

    Parameters
    ----------
    settings : dict
        Global configuration loaded from settings.yaml.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings: Dict[str, Any] = settings

        # Internal storage
        self._structures: Dict[str, Dict[str, Any]] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------
    # Structure storage
    # ------------------------------------------------------------
    def store_structure(self, file_path: Path, structure: Dict[str, Any]) -> None:
        """
        Store the AST structure for a given file.

        Parameters
        ----------
        file_path : Path
            File whose structure was extracted.

        structure : dict
            Structure dictionary produced by ASTInspector.
        """
        key = file_path.resolve().as_posix()
        self._structures[key] = structure

    def get_structure(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Retrieve the AST structure for a given file.

        Parameters
        ----------
        file_path : Path
            File whose structure should be retrieved.

        Returns
        -------
        Optional[dict]
            Structure dictionary or None if not found.
        """
        key = file_path.resolve().as_posix()
        return self._structures.get(key)

    # ------------------------------------------------------------
    # Schema storage
    # ------------------------------------------------------------
    def store_schema(self, file_path: Path, schema: Dict[str, Any]) -> None:
        """
        Store the inference schema for a given file.

        Parameters
        ----------
        file_path : Path
            File whose schema was produced.

        schema : dict
            Schema dictionary produced by SchemaBuilder.
        """
        key = file_path.resolve().as_posix()
        self._schemas[key] = schema

    def get_schema(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Retrieve the inference schema for a given file.

        Parameters
        ----------
        file_path : Path
            File whose schema should be retrieved.

        Returns
        -------
        Optional[dict]
            Schema dictionary or None if not found.
        """
        key = file_path.resolve().as_posix()
        return self._schemas.get(key)

    # ------------------------------------------------------------
    # Clearing
    # ------------------------------------------------------------
    def clear(self) -> None:
        """
        Clear all stored structures and schemas.
        """
        self._structures.clear()
        self._schemas.clear()

    # ------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------
    def list_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary of all files stored in the registry.

        Returns
        -------
        dict
            { file_path: structure_dict }
        """
        return dict(self._structures)

    def list_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary of all schemas stored in the registry.

        Returns
        -------
        dict
            { file_path: schema_dict }
        """
        return dict(self._schemas)
