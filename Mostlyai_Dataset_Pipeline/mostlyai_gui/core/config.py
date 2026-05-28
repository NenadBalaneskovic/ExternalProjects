"""
config.py

Configuration management for the Data Privacy Workbench GUI.

This module provides:
    - A ConfigLoader class that loads/saves a JSON config file
    - Automatic creation of the config directory
    - Safe defaults when no config exists
    - Logging integration
    - A clean API for reading/writing user preferences

The configuration file stores:
    - Default output paths
    - MOSTLY AI API key
    - UI preferences (optional)
    - Any future settings

Author: Nenad
Date: May 2026
"""

import json
import os
from pathlib import Path


# ----------------------------------------------------------------------
# Helper: determine config directory
# ----------------------------------------------------------------------

def get_config_dir() -> str:
    """
    Return the platform‑appropriate configuration directory.

    Windows:
        C:/Users/<User>/AppData/Roaming/data_privacy_workbench/

    Linux/macOS:
        ~/.config/data_privacy_workbench/
    """
    if os.name == "nt":  # Windows
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:  # Linux / macOS
        base = Path.home() / ".config"

    return str(base / "data_privacy_workbench")


# ----------------------------------------------------------------------
# ConfigLoader
# ----------------------------------------------------------------------

class ConfigLoader:
    """
    Load and save application configuration from config.json.

    The config file is stored in:
        <config_dir>/config.json

    Example structure:
    {
        "mostlyai_api_key": "",
        "default_cleaned_output": "output/cleaned",
        "default_anonymized_output": "output/anonymized",
        "default_pseudonymized_output": "output/pseudonymized",
        "default_synthetic_output": "output/synthetic",
        "default_mapping_output": "output/mappings"
    }
    """

    def __init__(self):
        self.config_dir = Path(get_config_dir())
        self.config_path = self.config_dir / "config.json"

        # Ensure directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------

    def load(self) -> dict:
        """
        Load configuration from config.json.
        If the file does not exist, return default config.
        """
        if not self.config_path.exists():
            return self._default_config()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # Corrupted config → fallback to defaults
            return self._default_config()

    # ------------------------------------------------------------------
    # Save configuration
    # ------------------------------------------------------------------

    def save(self, config: dict) -> None:
        """
        Save configuration to config.json.

        Parameters
        ----------
        config : dict
            Configuration dictionary to save.
        """
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration: {e}")

    # ------------------------------------------------------------------
    # Default configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _default_config() -> dict:
        """
        Return default configuration values.
        """
        return {
            "mostlyai_api_key": "",
            "default_cleaned_output": "output/cleaned",
            "default_anonymized_output": "output/anonymized",
            "default_pseudonymized_output": "output/pseudonymized",
            "default_synthetic_output": "output/synthetic",
            "default_mapping_output": "output/mappings",
        }
