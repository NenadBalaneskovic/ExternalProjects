"""
validators.py (GUI Layer)
-------------------------
Validation utilities for GUI input fields in the Slurm HPC–QPU Workflow
Orchestrator.

This module:
    - validates workflow file paths entered in the GUI
    - validates manual QPU credential fields
    - validates Slurm preview readiness
    - provides safe, deterministic checks
    - NEVER executes user workflow code

It complements core.validators but remains GUI‑specific.
"""

from pathlib import Path
from typing import Dict, Any


# ----------------------------------------------------------------------
# Basic Path Validators (GUI Layer)
# ----------------------------------------------------------------------

def is_valid_python_file(path: str | Path) -> bool:
    """
    Validate that the given path points to an existing Python file.
    GUI-safe: no execution, no side effects.
    """
    if not path:
        return False

    p = Path(path)
    return p.exists() and p.is_file() and p.suffix == ".py"


def is_valid_directory(path: str | Path) -> bool:
    """
    Validate that the given path points to an existing directory.
    """
    if not path:
        return False

    p = Path(path)
    return p.exists() and p.is_dir()


# ----------------------------------------------------------------------
# QPU Credential Validators (GUI Layer)
# ----------------------------------------------------------------------

def is_valid_api_key(api_key: str) -> bool:
    """
    Validate QPU API key.
    Accepts placeholders like {{API_KEY}}.
    """
    if not api_key:
        return False

    if api_key.startswith("{{") and api_key.endswith("}}"):
        return True

    return len(api_key.strip()) > 0


def is_valid_runtime_url(url: str) -> bool:
    """
    Validate QPU runtime URL.
    Accepts placeholders like {{RUNTIME_URL}}.
    """
    if not url:
        return False

    if url.startswith("{{") and url.endswith("}}"):
        return True

    return url.startswith("http")


# ----------------------------------------------------------------------
# Combined Credential Validation
# ----------------------------------------------------------------------

def validate_credentials(values: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate GUI credential fields.
    Returns a dict of error messages keyed by field name.
    """
    errors = {}

    if values.get("ENABLE_CREDS"):
        api_key = values.get("API_KEY", "")
        runtime_url = values.get("RUNTIME_URL", "")

        if not is_valid_api_key(api_key):
            errors["API_KEY"] = "Invalid API key."

        if not is_valid_runtime_url(runtime_url):
            errors["RUNTIME_URL"] = "Invalid runtime URL."

    return errors


# ----------------------------------------------------------------------
# Workflow Path Validation (GUI Layer)
# ----------------------------------------------------------------------

def validate_workflow_path(values: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate workflow file path from GUI input.
    """
    errors = {}
    wf = values.get("WORKFLOW_PATH", "")

    if not is_valid_python_file(wf):
        errors["WORKFLOW_PATH"] = "Invalid or missing workflow file."

    return errors


# ----------------------------------------------------------------------
# Full GUI Input Validation
# ----------------------------------------------------------------------

def validate_gui(values: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate all GUI inputs.
    Returns a dict of error messages keyed by field name.
    """
    errors = {}

    # Workflow file
    errors.update(validate_workflow_path(values))

    # Credentials
    errors.update(validate_credentials(values))

    return errors