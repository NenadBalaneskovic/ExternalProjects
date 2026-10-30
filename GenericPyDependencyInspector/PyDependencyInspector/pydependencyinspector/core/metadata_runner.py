"""
core/metadata_runner.py

Responsible for:
- Collecting metadata for the selected project.
- Returning a structured MetadataResult object.
- Replaces the old PyInstallerRunner (no subprocess, no PyInstaller).
- Backend-only (no GUI imports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Callable
import logging
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetadataResult:
    """
    Represents the outcome of a metadata collection run.

    Fields:
    - success: always True unless dependency resolver errors exist
    - warnings: non-fatal issues
    - errors: fatal issues (resolver errors)
    - duration_seconds: total metadata collection time
    """

    success: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# MetadataRunner – public API
# ---------------------------------------------------------------------------

class MetadataRunner:
    """
    Collects metadata for the project.

    Responsibilities:
    - Run dependency resolution (already done by caller)
    - Aggregate warnings/errors
    - Provide a structured MetadataResult
    - Stream logs to GUI log panel (optional)

    The GUI will call:
        runner = MetadataRunner()
        result = runner.run(log_callback)
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        resolution_warnings: List[str],
        resolution_errors: List[str],
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> MetadataResult:
        """
        Collect metadata and return a MetadataResult.

        :param resolution_warnings: warnings from dependency_resolver
        :param resolution_errors: errors from dependency_resolver
        :param log_callback: GUI log callback
        :return: MetadataResult
        """

        start_time = time.time()

        warnings = list(resolution_warnings)
        errors = list(resolution_errors)

        if log_callback:
            log_callback("[INFO] Collecting metadata...")
            if warnings:
                for w in warnings:
                    log_callback(f"[WARN] {w}")
            if errors:
                for e in errors:
                    log_callback(f"[ERR] {e}")

        # Metadata collection is always successful unless resolver errors exist
        success = len(errors) == 0

        duration = time.time() - start_time

        result = MetadataResult(
            success=success,
            warnings=warnings,
            errors=errors,
            duration_seconds=duration,
        )

        logger.info("Metadata collection finished in %.2f seconds", duration)
        return result


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_metadata(
    resolution_warnings: List[str],
    resolution_errors: List[str],
    log_callback: Optional[Callable[[str], None]] = None,
) -> MetadataResult:
    """
    Convenience wrapper for simple usage.

    Example:
        result = run_metadata(resolution_result.warnings, resolution_result.errors, print)
    """
    runner = MetadataRunner()
    return runner.run(resolution_warnings, resolution_errors, log_callback)
