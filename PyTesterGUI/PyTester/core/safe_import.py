"""
SafeImporter

This subsystem is responsible for:
- safely importing user-uploaded Python modules
- preventing dangerous side effects during import
- restricting builtins and globals
- enforcing a timeout for import operations
- returning a module object or None on failure

It is intentionally conservative:
no execution beyond import, no attribute calls, no dynamic behavior.
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Dict, Any, Optional, Callable


class SafeImporter:
    """
    Safely import user-uploaded Python modules.

    Parameters
    ----------
    settings : dict
        Global configuration loaded from settings.yaml.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings: Dict[str, Any] = settings
        self.timeout_seconds: int = settings["core"]["safe_import"]["timeout_seconds"]

    # ------------------------------------------------------------
    # Restricted builtins
    # ------------------------------------------------------------
    def _restricted_builtins(self) -> Dict[str, Any]:
        """
        Return a restricted set of builtins to prevent dangerous operations.

        Only safe introspection and basic types are allowed.
        """
        allowed = {
            "abs": builtins.abs,
            "len": builtins.len,
            "range": builtins.range,
            "enumerate": builtins.enumerate,
            "min": builtins.min,
            "max": builtins.max,
            "sum": builtins.sum,
            "print": builtins.print,
            "dict": builtins.dict,
            "list": builtins.list,
            "set": builtins.set,
            "tuple": builtins.tuple,
            "float": builtins.float,
            "int": builtins.int,
            "str": builtins.str,
            "bool": builtins.bool,
        }
        return allowed

    # ------------------------------------------------------------
    # Timeout wrapper
    # ------------------------------------------------------------
    def _run_with_timeout(self, func: Callable[[], ModuleType]) -> Optional[ModuleType]:
        """
        Run a function with a timeout using a thread.

        Parameters
        ----------
        func : Callable
            Function that performs the import.

        Returns
        -------
        Optional[ModuleType]
            Imported module or None if timeout or error occurs.
        """
        result: Dict[str, Optional[ModuleType]] = {"module": None}

        def target() -> None:
            try:
                result["module"] = func()
            except Exception:
                result["module"] = None

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(self.timeout_seconds)

        if thread.is_alive():
            return None

        return result["module"]

    # ------------------------------------------------------------
    # Safe import
    # ------------------------------------------------------------
    def import_file(self, file_path: Path) -> Optional[ModuleType]:
        """
        Safely import a Python file as a module.

        Parameters
        ----------
        file_path : Path
            Path to the Python file to import.

        Returns
        -------
        Optional[ModuleType]
            Imported module or None if import fails.
        """
        if not file_path.exists():
            return None

        def do_import() -> ModuleType:
            # Create a module spec
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                raise ImportError("Could not create module spec.")

            # Create a new module object
            module = importlib.util.module_from_spec(spec)

            # Restrict builtins
            module.__dict__["__builtins__"] = self._restricted_builtins()

            # Load the module
            spec.loader.exec_module(module)
            return module

        return self._run_with_timeout(do_import)
