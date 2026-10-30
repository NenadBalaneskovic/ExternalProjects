"""
CoverageRunner

This subsystem is responsible for:
- executing coverage.py on generated test files
- collecting structured coverage metrics
- running in a controlled subprocess environment
- remaining deterministic and side-effect-aware

It does not import user code directly; it delegates to coverage.py.
"""

from __future__ import annotations

import subprocess
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


class CoverageRunner:
    """
    Run coverage.py on generated test files and collect results.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.python_executable = settings["execution"]["coverage"]["python_executable"]
        self.max_duration_seconds = settings["execution"]["coverage"]["max_duration_seconds"]

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def run(self, test_files: Any = None, source_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run coverage.py on the given test files.
        """

        if test_files is None:
            return {
                "status": "error",
                "exit_code": -1,
                "stdout": "",
                "stderr": "[PyTester] CoverageRunner.run() requires at least one test file",
                "total_coverage": 0.0,
                "files": {},
            }

        # Normalize test_files
        if isinstance(test_files, Path):
            test_files = [test_files]
        elif not isinstance(test_files, list):
            return {
                "status": "error",
                "exit_code": -1,
                "stdout": "",
                "stderr": "[PyTester] invalid test_files argument",
                "total_coverage": 0.0,
                "files": {},
            }

        if not test_files:
            return {
                "status": "ok",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "total_coverage": 0.0,
                "files": {},
            }

        # ------------------------------------------------------------
        # Step 1: run coverage with pytest
        # ------------------------------------------------------------
        cmd_run = [
            self.python_executable,
            "-m",
            "coverage",
            "run",
        ]

        if isinstance(source_dir, Path):
            cmd_run.extend(["--source", str(source_dir)])

        cmd_run.extend(["-m", "pytest"])
        cmd_run.extend(str(p) for p in test_files)

        try:
            proc_run = subprocess.run(
                cmd_run,
                capture_output=True,
                text=True,
                timeout=self.max_duration_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "error",
                "exit_code": -1,
                "stdout": exc.stdout or "",
                "stderr": (exc.stderr or "") + "\n[PyTester] coverage run timed out",
                "total_coverage": 0.0,
                "files": {},
            }
        except Exception as exc:
            return {
                "status": "error",
                "exit_code": -1,
                "stdout": "",
                "stderr": f"[PyTester] coverage run failed to start: {exc}",
                "total_coverage": 0.0,
                "files": {},
            }

        # ------------------------------------------------------------
        # Step 2: generate coverage report
        # ------------------------------------------------------------
        cmd_report = [
            self.python_executable,
            "-m",
            "coverage",
            "report",
            "-m",
        ]

        try:
            proc_report = subprocess.run(
                cmd_report,
                capture_output=True,
                text=True,
                timeout=self.max_duration_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "error",
                "exit_code": -1,
                "stdout": exc.stdout or "",
                "stderr": (exc.stderr or "") + "\n[PyTester] coverage report timed out",
                "total_coverage": 0.0,
                "files": {},
            }
        except Exception as exc:
            return {
                "status": "error",
                "exit_code": -1,
                "stdout": "",
                "stderr": f"[PyTester] coverage report failed to start: {exc}",
                "total_coverage": 0.0,
                "files": {},
            }

        # ------------------------------------------------------------
        # Step 3: parse coverage output
        # ------------------------------------------------------------
        parsed = self._parse_report(proc_report.stdout)

        status = "ok" if proc_run.returncode == 0 else "error"

        return {
            "status": status,
            "exit_code": proc_run.returncode,
            "stdout": proc_report.stdout,
            "stderr": proc_report.stderr,
            "total_coverage": parsed["total_coverage"],
            "files": parsed["files"],
        }

    # ------------------------------------------------------------
    # Coverage report parsing
    # ------------------------------------------------------------
    def _parse_report(self, text: str) -> Dict[str, Any]:
        files: Dict[str, Any] = {}
        total_coverage: float = 0.0

        for line in text.splitlines():
            line = line.strip()
            if not line or "%" not in line:
                continue

            # Match coverage lines
            m = re.match(r"(\S+)\s+\d+\s+\d+\s+(\d+)%\s*(.*)", line)
            if not m:
                continue

            filename = m.group(1)
            coverage_pct = float(m.group(2))
            missing_raw = m.group(3).strip()

            # TOTAL line → global coverage
            if filename.upper() == "TOTAL":
                total_coverage = coverage_pct
                continue

            # Parse missing lines
            missing_lines = []
            if missing_raw:
                for part in missing_raw.split(","):
                    part = part.strip()
                    if "-" in part:
                        start, end = part.split("-")
                        missing_lines.extend(range(int(start), int(end) + 1))
                    else:
                        try:
                            missing_lines.append(int(part))
                        except ValueError:
                            pass

            files[filename] = {
                "coverage": coverage_pct,
                "missing": missing_lines,
            }

        return {
            "total_coverage": total_coverage,
            "files": files,
        }

    # ------------------------------------------------------------
    # Summary helper
    # ------------------------------------------------------------
    def summarize(self, result: Dict[str, Any]) -> str:
        lines = ["=== Coverage Summary ===", ""]
        lines.append(f"status: {result.get('status')}")
        lines.append(f"exit_code: {result.get('exit_code')}")
        lines.append(f"total_coverage: {result.get('total_coverage')}")
        lines.append("")
        lines.append("Files:")

        for fname, info in result.get("files", {}).items():
            lines.append(f"  {fname}: {info['coverage']}%")
            if info["missing"]:
                lines.append(f"    missing: {info['missing']}")
            else:
                lines.append("    missing: none")

        lines.append("")
        lines.append("stdout:")
        lines.append(result.get("stdout", ""))
        lines.append("")
        lines.append("stderr:")
        lines.append(result.get("stderr", ""))

        return "\n".join(lines)
