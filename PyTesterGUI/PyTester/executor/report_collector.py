"""
ReportCollector

This subsystem is responsible for:
- aggregating pytest results, coverage results, and logs
- producing a unified, deterministic execution report
- normalizing error messages and metadata
- remaining pure and side-effect-aware

It does not execute tests; it only collects and merges results.
"""

from __future__ import annotations

from typing import Dict, Any, Optional


class ReportCollector:
    """
    Collect and merge execution results into a unified report.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def collect(
        self,
        pytest_result: Optional[Dict[str, Any]] = None,
        coverage_result: Optional[Dict[str, Any]] = None,
        logs: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Merge pytest results, coverage results, and logs.
        """

        # Fail gracefully if missing inputs
        if pytest_result is None or coverage_result is None:
            return {
                "status": "error",
                "pytest": pytest_result or {},
                "coverage": coverage_result or {},
                "logs": logs,
                "summary": {
                    "exit_code": None,
                    "total_coverage": 0.0,
                    "tested_files": [],
                    "missing_lines": {},
                },
                "error": "[PyTester] ReportCollector.collect() requires pytest_result and coverage_result",
            }

        # Merge status (pytest OR coverage error → global error)
        status = self._merge_status(pytest_result, coverage_result)

        # Build unified summary
        summary = self._build_summary(pytest_result, coverage_result)

        # Unified report
        return {
            "status": status,

            "pytest": {
                "stdout": pytest_result.get("stdout", ""),
                "stderr": pytest_result.get("stderr", ""),
                "exit_code": pytest_result.get("exit_code", 0),
                "files": pytest_result.get("files", []),
                "durations": pytest_result.get("durations", {}),
                "test_results": pytest_result.get("test_results", {}),
                "failures": pytest_result.get("failures", []),
            },

            "coverage": {
                "status": coverage_result.get("status", "unknown"),
                "exit_code": coverage_result.get("exit_code", None),
                "total_coverage": float(coverage_result.get("total_coverage", 0.0)),
                "files": coverage_result.get("files", {}),
                "stdout": coverage_result.get("stdout", ""),
                "stderr": coverage_result.get("stderr", ""),
            },

            "logs": logs,
            "summary": summary,
        }

    # ------------------------------------------------------------
    # Status merging
    # ------------------------------------------------------------
    def _merge_status(
        self,
        pytest_result: Dict[str, Any],
        coverage_result: Dict[str, Any],
    ) -> str:
        if pytest_result.get("status") == "error":
            return "error"
        if coverage_result.get("status") == "error":
            return "error"
        return "ok"

    # ------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------
    def _build_summary(
        self,
        pytest_result: Dict[str, Any],
        coverage_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        tested_files = pytest_result.get("files", [])
        total_coverage = float(coverage_result.get("total_coverage", 0.0))

        # Extract missing lines (CoverageRunner already filters out TOTAL)
        missing_lines = {
            fname: info.get("missing", [])
            for fname, info in coverage_result.get("files", {}).items()
        }

        return {
            "exit_code": pytest_result.get("exit_code", 0),
            "total_coverage": total_coverage,
            "tested_files": tested_files,
            "missing_lines": missing_lines,
        }

    # ------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------
    def summarize(self, report: Dict[str, Any]) -> str:
        lines = ["=== Unified Execution Report ===", ""]
        lines.append(f"status: {report.get('status')}")
        lines.append("")

        summary = report.get("summary", {})
        lines.append(f"exit_code: {summary.get('exit_code')}")
        lines.append(f"total_coverage: {summary.get('total_coverage')}")
        lines.append("")

        lines.append("tested_files:")
        for f in summary.get("tested_files", []):
            lines.append(f"  - {f}")
        lines.append("")

        lines.append("missing_lines:")
        for fname, missing in summary.get("missing_lines", {}).items():
            if missing:
                lines.append(f"  {fname}: {missing}")
            else:
                lines.append(f"  {fname}: none")
        lines.append("")

        lines.append("pytest stdout:")
        lines.append(report.get("pytest", {}).get("stdout", ""))
        lines.append("")

        lines.append("pytest stderr:")
        lines.append(report.get("pytest", {}).get("stderr", ""))
        lines.append("")

        lines.append("coverage stdout:")
        lines.append(report.get("coverage", {}).get("stdout", ""))
        lines.append("")

        lines.append("coverage stderr:")
        lines.append(report.get("coverage", {}).get("stderr", ""))
        lines.append("")

        if report.get("logs"):
            lines.append("logs:")
            lines.append(report.get("logs"))
            lines.append("")

        return "\n".join(lines)
