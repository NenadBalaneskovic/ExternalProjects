"""
utils/wheel_inspector.py

Responsible for:
- Inspecting Python wheel (.whl) filenames.
- Extracting Python version tags, ABI tags, and platform tags.
- Determining compatibility with the current OS and architecture.
- Providing a unified API for dependency resolution and offline wheel workflows.

This module is intentionally:
- Pure Python.
- GUI-agnostic.
- Safe in offline environments.
- Compatible with PEP 425 wheel filename conventions.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

from .os_detection import detect_os, OSInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WheelInfo:
    """
    Represents parsed metadata from a wheel filename.

    Fields:
    - name: package name
    - version: package version
    - python_tag: e.g. "cp311", "py3"
    - abi_tag: e.g. "cp311", "none"
    - platform_tag: e.g. "win_amd64", "manylinux2014_x86_64"
    - filename: original wheel filename
    """
    name: str
    version: str
    python_tag: str
    abi_tag: str
    platform_tag: str
    filename: str


@dataclass(frozen=True)
class WheelCompatibility:
    """
    Represents compatibility of a wheel with the current environment.

    Fields:
    - is_compatible: True if wheel matches OS + architecture
    - reason: explanation string
    """
    is_compatible: bool
    reason: str


# ---------------------------------------------------------------------------
# WheelInspector – public API
# ---------------------------------------------------------------------------

class WheelInspector:
    """
    Parses wheel filenames and determines compatibility with the current OS.

    Responsibilities:
    - Parse wheel filenames according to PEP 425.
    - Extract python, ABI, and platform tags.
    - Compare platform tags with OSInfo.
    - Provide a unified API for dependency resolution and offline builds.

    Example:
        inspector = WheelInspector()
        info = inspector.parse("numpy-1.26.4-cp311-cp311-win_amd64.whl")
        compat = inspector.check_compatibility(info)
    """

    WHEEL_REGEX = re.compile(
        r"^(?P<name>.+)-(?P<version>[^-]+)-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^.]+)\.whl$"
    )

    def __init__(self, os_info: Optional[OSInfo] = None) -> None:
        self.os_info = os_info or detect_os()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def parse(self, filename: str) -> Optional[WheelInfo]:
        """
        Parse a wheel filename into WheelInfo.

        :param filename: e.g. "pandas-2.2.1-cp311-cp311-win_amd64.whl"
        :return: WheelInfo or None if parsing fails
        """
        match = self.WHEEL_REGEX.match(filename)
        if not match:
            logger.warning("Invalid wheel filename format: %s", filename)
            return None

        info = WheelInfo(
            name=match.group("name"),
            version=match.group("version"),
            python_tag=match.group("python"),
            abi_tag=match.group("abi"),
            platform_tag=match.group("platform"),
            filename=filename,
        )

        logger.debug("Parsed wheel: %s", info)
        return info

    def check_compatibility(self, wheel: WheelInfo) -> WheelCompatibility:
        """
        Determine whether a wheel is compatible with the current OS + architecture.

        :param wheel: WheelInfo
        :return: WheelCompatibility
        """
        os_name = self.os_info.os_name
        arch = self.os_info.architecture

        platform_tag = wheel.platform_tag.lower()

        # Universal wheels
        if platform_tag in ("any", "none-any"):
            return WheelCompatibility(True, "Universal wheel")

        # Windows
        if os_name == "windows":
            if "win" in platform_tag and arch in platform_tag:
                return WheelCompatibility(True, "Matches Windows architecture")
            return WheelCompatibility(False, f"Wheel is for '{platform_tag}', not Windows {arch}")

        # Linux
        if os_name == "linux":
            if "manylinux" in platform_tag and arch in platform_tag:
                return WheelCompatibility(True, "Matches manylinux architecture")
            if "linux" in platform_tag and arch in platform_tag:
                return WheelCompatibility(True, "Matches Linux architecture")
            return WheelCompatibility(False, f"Wheel is for '{platform_tag}', not Linux {arch}")

        # macOS
        if os_name == "macos":
            if "macos" in platform_tag and arch in platform_tag:
                return WheelCompatibility(True, "Matches macOS architecture")
            return WheelCompatibility(False, f"Wheel is for '{platform_tag}', not macOS {arch}")

        # Unknown OS
        return WheelCompatibility(False, f"Unknown OS '{os_name}'")

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #

    def is_compatible(self, filename: str) -> WheelCompatibility:
        """
        Parse + check compatibility in one step.

        :param filename: wheel filename
        :return: WheelCompatibility
        """
        info = self.parse(filename)
        if info is None:
            return WheelCompatibility(False, "Invalid wheel filename")
        return self.check_compatibility(info)
