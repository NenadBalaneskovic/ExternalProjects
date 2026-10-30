"""
InputLoader

This subsystem is responsible for:
- loading Python source files from disk
- validating file existence
- returning file contents as text
- providing a clean interface for upstream modules

It is intentionally minimal:
no parsing, no AST logic, no syntax checking.
Those responsibilities belong to other subsystems.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional


class InputLoader:
    """
    Load Python source files from disk.

    Parameters
    ----------
    settings : dict
        Global configuration loaded from settings.yaml.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings: Dict[str, Any] = settings

    # ------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------
    def load(self, file_path: Path) -> Optional[str]:
        """
        Load a Python file and return its contents as a string.

        Parameters
        ----------
        file_path : Path
            Path to the Python file to load.

        Returns
        -------
        Optional[str]
            File contents as a string, or None if the file does not exist.
        """
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    # ------------------------------------------------------------
    # Workspace helper
    # ------------------------------------------------------------
    def load_from_workspace(self, filename: str) -> Optional[str]:
        """
        Load a file from workspace/uploaded_files.

        Parameters
        ----------
        filename : str
            Name of the file inside workspace/uploaded_files.

        Returns
        -------
        Optional[str]
            File contents or None if not found.
        """
        upload_dir = Path(self.settings["paths"]["uploaded_files"])
        file_path = upload_dir / filename
        return self.load(file_path)
