"""
core/build_report.py

Responsible for:
- Generating a structured dependency summary report.
- Combining dependency information and environment metadata.
- Rendering the report as HTML (for GUI display).
- Returning a structured BuildReport object.

This module is intentionally:
- GUI-agnostic.
- Pure Python.
- Compatible with dependency_resolver and future metadata runners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import datetime
import html
import logging

from .dependency_resolver import DependencyNode, ResolutionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BuildReportDependency:
    name: str
    version: str
    type: str
    summary: str


@dataclass
class BuildReport:
    """
    Structured dependency summary consumed by GUI.
    """

    project_name: str
    version: Optional[str]
    os_profile: str

    status: str
    generated_at: str

    dependencies: List[BuildReportDependency] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    html: str = ""  # fragment HTML

    def is_successful(self) -> bool:
        return self.status.lower() == "successful" and not self.errors


# ---------------------------------------------------------------------------
# BuildReportGenerator
# ---------------------------------------------------------------------------

class BuildReportGenerator:
    """
    Generates a BuildReport object and fragment HTML.
    PyInstaller is no longer used; this is purely a dependency summary.
    """

    def generate(
        self,
        project_name: str,
        os_profile: str,
        resolution_result: ResolutionResult,
    ) -> BuildReport:

        logger.info("Generating dependency summary for '%s' (%s)", project_name, os_profile)

        version = resolution_result.root.version if resolution_result.root else None
        status = "Successful"  # always successful unless resolver errors exist
        generated_at = datetime.datetime.utcnow().isoformat() + "Z"

        deps = [
            BuildReportDependency(
                name=node.name,
                version=node.version,
                type=node.dep_type.value,
                summary=node.extras.get("summary", ""),
            )
            for node in resolution_result.flat_list
        ]

        warnings = list(resolution_result.warnings)
        errors = list(resolution_result.errors)

        report = BuildReport(
            project_name=project_name,
            version=version,
            os_profile=os_profile,
            status=status,
            generated_at=generated_at,
            dependencies=deps,
            warnings=warnings,
            errors=errors,
        )

        report.html = self._render_html(report)
        return report

    # ------------------------------------------------------------------ #
    # Fragment HTML renderer
    # ------------------------------------------------------------------ #

    def _render_html(self, report: BuildReport) -> str:
        """
        Produce fragment HTML (no <html>, <head>, <body>, <style>).
        Suitable for QTextBrowser.
        """

        esc = html.escape

        deps_rows = "".join(
            f"<tr>"
            f"<td>{esc(dep.name)}</td>"
            f"<td>{esc(dep.version)}</td>"
            f"<td>{esc(dep.type)}</td>"
            f"<td>{esc(dep.summary)}</td>"
            f"</tr>"
            for dep in report.dependencies
        )

        warnings_list = (
            "".join(f"<li>{esc(w)}</li>" for w in report.warnings)
            if report.warnings else "<li><em>None</em></li>"
        )

        errors_list = (
            "".join(f"<li>{esc(e)}</li>" for e in report.errors)
            if report.errors else "<li><em>None</em></li>"
        )

        return f"""
            <h2>Dependency Summary — {esc(report.project_name)}</h2>

            <p><strong>Version:</strong> {esc(report.version or "Unknown")}</p>
            <p><strong>OS Profile:</strong> {esc(report.os_profile)}</p>
            <p><strong>Generated at:</strong> {esc(report.generated_at)}</p>

            <hr>

            <h3>Dependencies</h3>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Version</th>
                    <th>Type</th>
                    <th>Summary</th>
                </tr>
                {deps_rows}
            </table>

            <hr>

            <h3>Warnings</h3>
            <ul>{warnings_list}</ul>

            <h3>Errors</h3>
            <ul>{errors_list}</ul>
        """


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_build_report(
    project_name: str,
    os_profile: str,
    resolution_result: ResolutionResult,
) -> BuildReport:
    generator = BuildReportGenerator()
    return generator.generate(project_name, os_profile, resolution_result)
