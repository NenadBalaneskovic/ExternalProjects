"""
core/requirements_exporter.py

Responsible for:
- Exporting a list of resolved dependencies into a requirements.txt file.
- Supporting pinned versions, optional hash mode, and custom formatting.
- Operating deterministically and safely in offline environments.
- Returning a structured RequirementsExportResult object.

This module is intentionally:
- GUI-agnostic.
- Pure Python.
- Compatible with the DependencyResolver output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import os
import logging

from .dependency_resolver import DependencyNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RequirementsExportResult:
    """
    Represents the outcome of exporting a requirements.txt file.

    Fields:
    - success: True if file was written successfully
    - output_path: path to the generated requirements.txt
    - warnings: non-fatal issues (missing versions, skipped packages)
    - errors: fatal issues (cannot write file)
    """
    success: bool
    output_path: str
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def is_successful(self) -> bool:
        return self.success and not self.errors


# ---------------------------------------------------------------------------
# RequirementsExporter – public API
# ---------------------------------------------------------------------------

class RequirementsExporter:
    """
    Exports a list of DependencyNode objects into a requirements.txt file.

    Responsibilities:
    - Convert dependency nodes into pip-compatible requirement lines.
    - Support pinned versions (default).
    - Support optional hash mode (future extension).
    - Write the file to disk.
    - Return a RequirementsExportResult.

    The GUI will call:
        exporter = RequirementsExporter()
        result = exporter.export(flat_list, "build/requirements.txt")
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def export(
        self,
        dependencies: List[DependencyNode],
        output_path: str,
        include_versions: bool = True,
    ) -> RequirementsExportResult:
        """
        Export dependencies to a requirements.txt file.

        :param dependencies: List of DependencyNode objects (flat list)
        :param output_path: Path to write requirements.txt
        :param include_versions: Whether to pin versions (default: True)
        :return: RequirementsExportResult
        """
        warnings: List[str] = []
        errors: List[str] = []

        # Step 1: Convert nodes to requirement lines
        lines = []
        for node in dependencies:
            if not node.name:
                warnings.append("Encountered dependency with missing name.")
                continue

            if include_versions:
                if node.version:
                    lines.append(f"{node.name}=={node.version}")
                else:
                    warnings.append(f"Missing version for '{node.name}', using unpinned format.")
                    lines.append(node.name)
            else:
                lines.append(node.name)

        # Step 2: Sort for determinism
        lines = sorted(set(lines), key=str.lower)

        # Step 3: Write file
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as exc:
            msg = f"Failed to write requirements file: {exc}"
            logger.exception(msg)
            errors.append(msg)
            return RequirementsExportResult(
                success=False,
                output_path=output_path,
                warnings=warnings,
                errors=errors,
            )

        logger.info("requirements.txt exported: %s", output_path)
        return RequirementsExportResult(
            success=True,
            output_path=output_path,
            warnings=warnings,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def export_requirements(
    dependencies: List[DependencyNode],
    output_path: str,
    include_versions: bool = True,
) -> RequirementsExportResult:
    """
    Convenience wrapper for simple usage.

    Example:
        result = export_requirements(flat_list, "build/requirements.txt")
    """
    exporter = RequirementsExporter()
    return exporter.export(dependencies, output_path, include_versions)
