"""
PytestRunner

This subsystem is responsible for:
- executing generated pytest test files
- collecting results in a deterministic, structured format
- running in a controlled subprocess environment
- remaining side-effect-aware but constrained

It does not import user code directly; it delegates to pytest as a subprocess.
"""

from __future__ import annotations

import subprocess
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


class PytestRunner:
    """
    Run pytest on generated test files and collect results.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.python_executable = settings["execution"]["pytest"]["python_executable"]
        self.max_duration_seconds = settings["execution"]["pytest"]["max_duration_seconds"]

    # ------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------
    def run(self, test_files: Any) -> Dict[str, Any]:
        """
        Run pytest on the given test files.
        """

        # Normalize input
        if isinstance(test_files, Path):
            test_files = [test_files]
        elif not isinstance(test_files, list):
            result = self._error_result("[PyTester] invalid test_files argument")
            self._write_json_report(result)
            return result

        if not test_files:
            result = self._empty_result()
            self._write_json_report(result)
            return result

        # ------------------------------------------------------------
        # Add workspace/source to PYTHONPATH
        # ------------------------------------------------------------
        source_dir = Path(self.settings["paths"]["source"]).resolve()

        env = os.environ.copy()
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(source_dir)
            if existing_path == ""
            else existing_path + os.pathsep + str(source_dir)
        )

        # ------------------------------------------------------------
        # Run pytest
        # ------------------------------------------------------------
        cmd_pytest = [self.python_executable, "-m", "pytest"]
        cmd_pytest.extend(str(p) for p in test_files)

        try:
            start = time.time()
            proc_pytest = subprocess.run(
                cmd_pytest,
                capture_output=True,
                text=True,
                timeout=self.max_duration_seconds,
                env=env,
            )
            end = time.time()
            duration_pytest = end - start
        except subprocess.TimeoutExpired as exc:
            result = self._timeout_result(exc, test_files)
            self._write_json_report(result)
            return result
        except Exception as exc:
            result = self._startup_error_result(exc, test_files)
            self._write_json_report(result)
            return result

        status = "ok" if proc_pytest.returncode == 0 else "error"

        result = {
            "status": status,
            "exit_code": proc_pytest.returncode,
            "stdout": proc_pytest.stdout,
            "stderr": proc_pytest.stderr,
            "files": [str(p) for p in test_files],
        }

        # ------------------------------------------------------------
        # Durations + pass/fail
        # ------------------------------------------------------------
        result["durations"] = {
            "pytest": duration_pytest,
            "coverage": 0.0,
            "total": duration_pytest,
        }

        passed, failed = self._extract_pass_fail(proc_pytest.stdout)
        result["test_results"] = {
            "passed": passed,
            "failed": failed,
        }

        result["failures"] = self._extract_failures(proc_pytest.stdout)

        # ------------------------------------------------------------
        # Write JSON report (pytest only)
        # ------------------------------------------------------------
        self._write_json_report(result)

        return result

    # ------------------------------------------------------------
    # JSON report writer
    # ------------------------------------------------------------
    def _write_json_report(self, result: Dict[str, Any]) -> None:
        report_dir = Path(self.settings["paths"]["test_reports"])
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / "pytest_report.json"

        payload = {
            "status": result.get("status"),
            "exit_code": result.get("exit_code"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "files": result.get("files", []),

            "durations": result.get("durations", {
                "pytest": 0.0,
                "coverage": 0.0,
                "total": 0.0,
            }),

            "test_results": result.get("test_results", {
                "passed": 0,
                "failed": 0,
            }),

            "failures": result.get("failures", []),
        }

        try:
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------
    # Helper result builders
    # ------------------------------------------------------------
    def _error_result(self, msg: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "exit_code": -1,
            "stdout": "",
            "stderr": msg,
            "files": [],
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "files": [],
        }

    def _timeout_result(self, exc, test_files) -> Dict[str, Any]:
        return {
            "status": "error",
            "exit_code": -1,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\n[PyTester] pytest timed out",
            "files": [str(p) for p in test_files],
            "durations": {
                "pytest": self.max_duration_seconds,
                "coverage": 0.0,
                "total": self.max_duration_seconds,
            },
        }

    def _startup_error_result(self, exc, test_files) -> Dict[str, Any]:
        return {
            "status": "error",
            "exit_code": -1,
            "stdout": "",
            "stderr": f"[PyTester] pytest failed to start: {exc}",
            "files": [str(p) for p in test_files],
            "durations": {
                "pytest": 0.0,
                "coverage": 0.0,
                "total": 0.0,
            },
        }

    # ------------------------------------------------------------
    # Pass/Fail extraction
    # ------------------------------------------------------------
    def _extract_pass_fail(self, stdout: str) -> (int, int):
        passed = 0
        failed = 0

        if not stdout:
            return passed, failed

        for line in stdout.splitlines():
            line = line.strip()
            if "passed" in line or "failed" in line:
                parts = line.replace(",", "").split()
                for i, p in enumerate(parts):
                    if p == "passed":
                        try:
                            passed = int(parts[i - 1])
                        except Exception:
                            pass
                    if p == "failed":
                        try:
                            failed = int(parts[i - 1])
                        except Exception:
                            pass

        return passed, failed

    # ------------------------------------------------------------
    # Failure extraction
    # ------------------------------------------------------------
    def _extract_failures(self, stdout: str) -> List[str]:
        failures: List[str] = []

        if not stdout:
            return failures

        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("FAILED") and "::" in line:
                failures.append(line)

        return failures

    # ------------------------------------------------------------
    # Summary helper
    # ------------------------------------------------------------
    def summarize(self, result: Dict[str, Any]) -> str:
        lines = ["=== Pytest Execution Summary ===", ""]
        lines.append(f"status: {result.get('status')}")
        lines.append(f"exit_code: {result.get('exit_code')}")
        lines.append(f"files: {', '.join(result.get('files', []))}")
        lines.append("")
        lines.append("stdout:")
        lines.append(result.get("stdout", ""))
        lines.append("")
        lines.append("stderr:")
        lines.append(result.get("stderr", ""))
        return "\n".join(lines)

