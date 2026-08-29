"""
core/dependency_resolver.py

Responsible for:
- Resolving dependencies for a given Python package name.
- Building a structured dependency tree suitable for GUI display.
- Providing OS-aware metadata (e.g. wheel compatibility hints).
- Emitting a high-level result object that other modules (GUI, build, export)
  can consume without knowing about pip internals.

This module is intentionally written to be:
- Pure Python (no heavy external dependencies).
- Backend-agnostic (can later switch from subprocess-based pip calls to
  a dedicated resolver library without breaking the public API).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Iterable, Tuple
import subprocess
import sys
import json
import logging
import re
import shutil
import textwrap


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

class DependencyType(Enum):
    """High-level classification of a dependency."""
    PYTHON_PACKAGE = "python_package"
    NATIVE_LIBRARY = "native_library"
    UNKNOWN = "unknown"


@dataclass
class DependencyNode:
    """
    Represents a single node in the dependency tree.

    This is the core structure that the GUI will consume:
    - name: package or library name
    - version: resolved version (if known)
    - dep_type: high-level type (Python package, native library, etc.)
    - children: nested dependencies
    - extras: arbitrary metadata (wheel tags, license, summary, etc.)
    """
    name: str
    version: Optional[str] = None
    dep_type: DependencyType = DependencyType.PYTHON_PACKAGE
    children: List["DependencyNode"] = field(default_factory=list)
    extras: Dict[str, str] = field(default_factory=dict)

    def add_child(self, child: "DependencyNode") -> None:
        self.children.append(child)

    def to_dict(self) -> Dict:
        """Convert to a JSON-serializable dict (useful for debugging/export)."""
        return {
            "name": self.name,
            "version": self.version,
            "dep_type": self.dep_type.value,
            "extras": self.extras,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class ResolutionResult:
    """
    High-level result of a dependency resolution run.

    - root: the root dependency node (the requested package)
    - flat_list: a flattened list of all dependencies (for quick lookup/export)
    - warnings: non-fatal issues encountered during resolution
    - errors: fatal issues (e.g. package not found)
    """
    root: Optional[DependencyNode]
    flat_list: List[DependencyNode]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def is_successful(self) -> bool:
        return self.root is not None and not self.errors


# ---------------------------------------------------------------------------
# OS profile abstraction
# ---------------------------------------------------------------------------

class OSProfile(Enum):
    """Target OS profile for resolution/build hints."""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "OSProfile":
        v = value.lower()
        if "win" in v:
            return cls.WINDOWS
        if "ubuntu" in v or "linux" in v:
            return cls.LINUX
        if "mac" in v or "darwin" in v:
            return cls.MACOS
        return cls.UNKNOWN


# ---------------------------------------------------------------------------
# DependencyResolver – public API
# ---------------------------------------------------------------------------

class DependencyResolver:
    """
    Main entry point for resolving dependencies of a Python package.

    Responsibilities:
    - Use pip (via subprocess) to inspect dependencies.
    - Optionally query PyPI JSON API for additional metadata.
    - Build a DependencyNode tree.
    - Provide OS-aware hints (e.g. wheel tags, native libs).

    This class is intentionally written so that:
    - The GUI only needs to call `resolve(package_name, os_profile_str)`.
    - The rest of the project can treat this as a black box.
    """

    def __init__(self, python_executable: Optional[str] = None) -> None:
        """
        :param python_executable:
            Path to the Python interpreter to use for pip calls.
            If None, uses the current interpreter (sys.executable).
        """
        self.python_executable = python_executable or sys.executable

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def resolve(self, package_name: str, os_profile_str: str) -> ResolutionResult:
        """
        Resolve dependencies for the given package name and target OS profile.

        This is the main method the GUI will call.

        :param package_name: Name of the top-level package (e.g. "pandas").
        :param os_profile_str: Human-readable OS string (e.g. "Windows 11").
        :return: ResolutionResult with a dependency tree and metadata.
        """
        logger.info("Starting dependency resolution for package '%s' (OS: %s)",
                    package_name, os_profile_str)

        os_profile = OSProfile.from_string(os_profile_str)
        warnings: List[str] = []
        errors: List[str] = []

        if not self._pip_available():
            msg = "pip is not available in the current environment."
            logger.error(msg)
            return ResolutionResult(root=None, flat_list=[], warnings=[], errors=[msg])

        # Step 1: Try to get basic info and dependencies via `pip show`.
        pkg_info, show_warnings, show_errors = self._pip_show(package_name)
        warnings.extend(show_warnings)
        errors.extend(show_errors)

        if pkg_info is None:
            msg = f"Package '{package_name}' could not be resolved via 'pip show'."
            logger.error(msg)
            errors.append(msg)
            return ResolutionResult(root=None, flat_list=[], warnings=warnings, errors=errors)

        # Step 2: Build a flat dependency mapping using `pip show` recursively.
        dep_map, dep_warnings = self._build_dependency_map(package_name)
        warnings.extend(dep_warnings)

        # Step 3: Build a tree from the flat map.
        root_node, flat_list = self._build_tree(package_name, dep_map, os_profile)

        # Step 4: Optionally enrich with PyPI metadata (summary, homepage, etc.).
        self._enrich_with_pypi_metadata(flat_list, warnings)

        logger.info("Dependency resolution completed for '%s'.", package_name)
        return ResolutionResult(root=root_node, flat_list=flat_list,
                                warnings=warnings, errors=errors)

    # ------------------------------------------------------------------ #
    # Internal helpers – pip interaction
    # ------------------------------------------------------------------ #

    def _pip_available(self) -> bool:
        """Check if pip is available for the configured Python interpreter."""
        if not shutil.which(self.python_executable):
            return False
        try:
            subprocess.run(
                [self.python_executable, "-m", "pip", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except Exception:
            return False

    def _pip_show(self, package_name: str) -> Tuple[Optional[Dict[str, str]], List[str], List[str]]:
        """
        Call `pip show <package>` and parse the output into a dict.

        Returns:
            (info_dict or None, warnings, errors)
        """
        warnings: List[str] = []
        errors: List[str] = []

        try:
            result = subprocess.run(
                [self.python_executable, "-m", "pip", "show", package_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except Exception as exc:
            msg = f"Failed to run 'pip show {package_name}': {exc}"
            logger.exception(msg)
            errors.append(msg)
            return None, warnings, errors

        if result.returncode != 0 or not result.stdout.strip():
            msg = f"'pip show {package_name}' returned no information."
            logger.warning(msg)
            warnings.append(msg)
            return None, warnings, errors

        info: Dict[str, str] = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()

        return info, warnings, errors

    def _build_dependency_map(self, root_package: str) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
        """
        Build a flat mapping of package -> pip show info by recursively
        following 'Requires' fields.

        Returns:
            (dep_map, warnings)
        """
        warnings: List[str] = []
        dep_map: Dict[str, Dict[str, str]] = {}

        visited: set[str] = set()
        to_visit: List[str] = [root_package]

        while to_visit:
            current = to_visit.pop()
            if current.lower() in visited:
                continue
            visited.add(current.lower())

            info, show_warnings, _ = self._pip_show(current)
            warnings.extend(show_warnings)

            if info is None:
                msg = f"Could not retrieve info for dependency '{current}'."
                logger.warning(msg)
                warnings.append(msg)
                continue

            dep_map[current] = info

            requires_str = info.get("Requires", "")
            requires = [r.strip() for r in requires_str.split(",") if r.strip()]
            for r in requires:
                if r.lower() not in visited:
                    to_visit.append(r)

        return dep_map, warnings

    # ------------------------------------------------------------------ #
    # Internal helpers – tree building & enrichment
    # ------------------------------------------------------------------ #

    def _build_tree(
        self,
        root_package: str,
        dep_map: Dict[str, Dict[str, str]],
        os_profile: OSProfile,
    ) -> Tuple[DependencyNode, List[DependencyNode]]:
        """
        Build a DependencyNode tree from the flat dep_map.

        Returns:
            (root_node, flat_list)
        """
        flat_list: List[DependencyNode] = []
        node_cache: Dict[str, DependencyNode] = {}

        def get_or_create_node(pkg_name: str) -> DependencyNode:
            key = pkg_name.lower()
            if key in node_cache:
                return node_cache[key]

            info = dep_map.get(pkg_name, {})
            version = info.get("Version")
            node = DependencyNode(
                name=pkg_name,
                version=version,
                dep_type=DependencyType.PYTHON_PACKAGE,
                extras={
                    "summary": info.get("Summary", ""),
                    "home_page": info.get("Home-page", ""),
                    "license": info.get("License", ""),
                    "location": info.get("Location", ""),
                    "requires": info.get("Requires", ""),
                    "os_profile": os_profile.value,
                },
            )
            node_cache[key] = node
            flat_list.append(node)
            return node

        # Build tree recursively
        def build_children(pkg_name: str) -> DependencyNode:
            node = get_or_create_node(pkg_name)
            info = dep_map.get(pkg_name, {})
            requires_str = info.get("Requires", "")
            requires = [r.strip() for r in requires_str.split(",") if r.strip()]

            for child_name in requires:
                child_node = build_children(child_name)
                if child_node not in node.children:
                    node.add_child(child_node)

            return node

        root_node = build_children(root_package)
        return root_node, flat_list

    def _enrich_with_pypi_metadata(
        self,
        nodes: Iterable[DependencyNode],
        warnings: List[str],
    ) -> None:
        """
        Optionally enrich nodes with PyPI JSON metadata.

        For now, this is implemented as a no-op placeholder that can be
        extended later. The public API is already in place so that the
        rest of the system does not need to change when enrichment is added.
        """
        # Placeholder: in a later iteration, we can:
        # - call `https://pypi.org/pypi/<name>/json`
        # - extract long description, project URLs, classifiers, etc.
        # For now, we keep this as a stub to preserve offline behavior.
        _ = nodes  # avoid unused variable warning
        _ = warnings
        # Example of how a warning could be added in the future:
        # warnings.append("PyPI metadata enrichment not yet implemented.")


# ---------------------------------------------------------------------------
# Convenience function for quick, non-GUI usage (e.g. tests, CLI)
# ---------------------------------------------------------------------------

def resolve_dependencies(package_name: str, os_profile_str: str) -> ResolutionResult:
    """
    Convenience wrapper around DependencyResolver for simple usage.

    Example:
        result = resolve_dependencies("pandas", "Windows 11")
        if result.is_successful():
            for node in result.flat_list:
                print(node.name, node.version)
    """
    resolver = DependencyResolver()
    return resolver.resolve(package_name, os_profile_str)
