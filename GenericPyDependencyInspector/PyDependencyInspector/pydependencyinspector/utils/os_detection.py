"""
utils/os_detection.py

Responsible for:
- Detecting the current operating system and architecture.
- Normalizing OS names into canonical identifiers.
- Providing helper functions for OS-aware logic (wheel tags, PyInstaller flags).
- Operating purely in Python without external dependencies.

This module is intentionally:
- GUI-agnostic.
- Pure Python.
- Safe in offline environments.
"""

from __future__ import annotations

import platform
import sys
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OSInfo:
    """
    Represents normalized OS information.

    Fields:
    - os_name: canonical OS name ("windows", "linux", "macos", "unknown")
    - os_version: human-readable version string
    - architecture: CPU architecture ("x86_64", "arm64", etc.)
    - raw_platform: raw platform string from Python
    """
    os_name: str
    os_version: str
    architecture: str
    raw_platform: str


# ---------------------------------------------------------------------------
# OS detection helpers
# ---------------------------------------------------------------------------

def detect_os() -> OSInfo:
    """
    Detect the current operating system and return normalized OSInfo.

    Returns:
        OSInfo object with canonical OS name and architecture.
    """
    raw = platform.platform()
    system = platform.system().lower()
    arch = platform.machine().lower()

    # Normalize OS name
    if "windows" in system:
        os_name = "windows"
    elif "linux" in system:
        os_name = "linux"
    elif "darwin" in system or "mac" in system:
        os_name = "macos"
    else:
        os_name = "unknown"

    # Normalize architecture
    arch_map = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
        "x86": "x86",
        "i386": "x86",
        "i686": "x86",
    }
    architecture = arch_map.get(arch, arch)

    version = platform.version()

    info = OSInfo(
        os_name=os_name,
        os_version=version,
        architecture=architecture,
        raw_platform=raw,
    )

    logger.debug("Detected OS: %s", info)
    return info


def normalize_os_string(os_string: str) -> str:
    """
    Normalize a human-readable OS string into a canonical identifier.

    Examples:
        "Windows 11" -> "windows"
        "Ubuntu 22.04" -> "linux"
        "macOS Ventura" -> "macos"

    :param os_string: Human-readable OS name
    :return: canonical OS identifier
    """
    s = os_string.lower()

    if "win" in s:
        return "windows"
    if "linux" in s or "ubuntu" in s or "debian" in s:
        return "linux"
    if "mac" in s or "darwin" in s or "os x" in s:
        return "macos"

    return "unknown"


def is_windows() -> bool:
    """Return True if running on Windows."""
    return detect_os().os_name == "windows"


def is_linux() -> bool:
    """Return True if running on Linux."""
    return detect_os().os_name == "linux"


def is_macos() -> bool:
    """Return True if running on macOS."""
    return detect_os().os_name == "macos"


def get_architecture() -> str:
    """
    Return normalized CPU architecture.

    Examples:
        "x86_64"
        "arm64"
        "x86"
    """
    return detect_os().architecture
