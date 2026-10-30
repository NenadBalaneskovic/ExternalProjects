"""
config/paths.py

Responsible for:
- Loading defaults.json and settings.yaml.
- Merging them into a single resolved configuration dictionary.
- Normalizing all filesystem paths.
- Providing a central API for accessing configuration values.
- Ensuring deterministic, OS-safe path handling.

This module is intentionally:
- Pure Python.
- GUI-agnostic.
- Safe in offline environments.
"""

from __future__ import annotations

import json
import yaml
import os
import logging
from typing import Any, Dict

from ..utils.file_helpers import normalize_path, ensure_directory
from ..utils.os_detection import detect_os

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load JSON config '%s': %s", path, exc)
        return {}


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Failed to load YAML config '%s': %s", path, exc)
        return {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    Values in 'override' take precedence.
    """
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# ConfigManager – public API
# ---------------------------------------------------------------------------

class ConfigManager:
    """
    Loads and merges defaults.json + settings.yaml.
    Provides normalized paths and configuration values.

    Usage:
        config = ConfigManager()
        out_dir = config.project_output_dir
        os_profile = config.os_profile
    """

    def __init__(self) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.defaults_path = os.path.join(base_dir, "defaults.json")
        self.settings_path = os.path.join(base_dir, "settings.yaml")

        self.defaults = _load_json(self.defaults_path)
        self.settings = _load_yaml(self.settings_path)

        # Merge settings.yaml over defaults.json
        self.config = _deep_merge(self.defaults, self.settings)

        # Detect OS if needed
        self.os_info = detect_os()

        # Normalize all known paths
        self._normalize_paths()

    # ------------------------------------------------------------------ #
    # Path normalization
    # ------------------------------------------------------------------ #

    def _normalize_paths(self) -> None:
        """
        Normalize all filesystem paths in the config.
        """
        project = self.config.get("project", {})
        paths = self.config.get("paths", {})

        # Project directories
        for key in ("output_directory", "log_directory", "report_directory", "spec_directory"):
            if key in project:
                project[key] = normalize_path(project[key])

        # Paths section
        for key in ("temp_directory", "cache_directory"):
            if key in paths:
                paths[key] = normalize_path(paths[key])

        # Ensure directories exist
        for path in [
            project.get("output_directory"),
            project.get("log_directory"),
            project.get("report_directory"),
            project.get("spec_directory"),
            paths.get("temp_directory"),
            paths.get("cache_directory"),
        ]:
            if path:
                try:
                    ensure_directory(path)
                except Exception:
                    pass  # Directory creation errors are logged in file_helpers

    # ------------------------------------------------------------------ #
    # Public getters
    # ------------------------------------------------------------------ #

    @property
    def os_profile(self) -> str:
        """
        Return the OS profile from settings.yaml or auto-detected.
        """
        profile = self.config.get("project", {}).get("os_profile", "auto")
        if profile == "auto":
            return self.os_info.os_name
        return profile

    @property
    def project_output_dir(self) -> str:
        return self.config["project"]["output_directory"]

    @property
    def project_log_dir(self) -> str:
        return self.config["project"]["log_directory"]

    @property
    def project_report_dir(self) -> str:
        return self.config["project"]["report_directory"]

    @property
    def project_spec_dir(self) -> str:
        return self.config["project"]["spec_directory"]

    @property
    def temp_dir(self) -> str:
        return self.config["paths"]["temp_directory"]

    @property
    def cache_dir(self) -> str:
        return self.config["paths"]["cache_directory"]

    @property
    def python_interpreter(self) -> str:
        """
        Return the Python interpreter path.
        'auto' resolves to sys.executable.
        """
        interp = self.config["paths"].get("python_interpreter", "auto")
        if interp == "auto":
            return os.path.abspath(os.sys.executable)
        return normalize_path(interp)

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Generic nested getter.

        Example:
            config.get("pyinstaller", "clean_build")
        """
        node = self.config
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------

# Most modules will simply import this:
config = ConfigManager()
