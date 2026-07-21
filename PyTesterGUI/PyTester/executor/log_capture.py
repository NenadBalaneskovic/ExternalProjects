"""
LogCapture

This subsystem is responsible for:
- capturing stdout/stderr from subprocess executions
- optionally capturing Python logging output
- normalizing logs into a single deterministic string
- remaining pure and side-effect-aware

It does not execute tests; it only captures and merges logs.
"""

from __future__ import annotations

import io
import logging
from typing import Dict, Any, Optional


class LogCapture:
    """
    Capture and normalize logs from subprocesses and Python logging.

    Parameters
    ----------
    settings : dict
        Global configuration loaded from settings.yaml.
    """

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.capture_python_logs = settings["execution"]["logs"]["capture_python_logs"]

        # Internal buffers
        self._python_log_stream: Optional[io.StringIO] = None
        self._python_handler: Optional[logging.Handler] = None

        self._last_subprocess_logs: str = ""
        self._last_python_logs: str = ""
        self._last_unified_logs: str = ""

    # ------------------------------------------------------------
    # Python logging capture
    # ------------------------------------------------------------
    def start_python_capture(self) -> None:
        if not self.capture_python_logs:
            return

        self._python_log_stream = io.StringIO()
        self._python_handler = logging.StreamHandler(self._python_log_stream)
        self._python_handler.setLevel(logging.DEBUG)

        root = logging.getLogger()
        root.addHandler(self._python_handler)

    def stop_python_capture(self) -> str:
        if not self.capture_python_logs:
            return ""

        root = logging.getLogger()
        if self._python_handler:
            root.removeHandler(self._python_handler)

        if self._python_log_stream:
            self._last_python_logs = self._python_log_stream.getvalue()
            return self._last_python_logs

        return ""

    # ------------------------------------------------------------
    # Subprocess log capture
    # ------------------------------------------------------------
    def capture_subprocess_logs(self, stdout: Optional[str], stderr: Optional[str]) -> str:
        out = stdout or ""
        err = stderr or ""

        if not out and not err:
            self._last_subprocess_logs = ""
            return ""

        lines = ["=== Subprocess Logs ===", ""]
        if out:
            lines.append("stdout:")
            lines.append(out)
            lines.append("")
        if err:
            lines.append("stderr:")
            lines.append(err)
            lines.append("")

        self._last_subprocess_logs = "\n".join(lines)
        return self._last_subprocess_logs

    # ------------------------------------------------------------
    # Unified log merging
    # ------------------------------------------------------------
    def merge(self, subprocess_logs: str, python_logs: str) -> str:
        if not subprocess_logs and not python_logs:
            self._last_unified_logs = ""
            return ""

        lines = ["=== Unified Logs ===", ""]

        if subprocess_logs:
            lines.append(subprocess_logs)
            lines.append("")

        if python_logs:
            lines.append("=== Python Logs ===")
            lines.append(python_logs)
            lines.append("")

        self._last_unified_logs = "\n".join(lines)
        return self._last_unified_logs

    # ------------------------------------------------------------
    # NEW: read_logs() for ExecutionPanel
    # ------------------------------------------------------------
    def read_logs(self) -> str:
        """
        Return the last unified logs captured.
        ExecutionPanel expects this method.
        """
        return self._last_unified_logs

    # ------------------------------------------------------------
    # Convenience helper
    # ------------------------------------------------------------
    def summarize(self, logs: str) -> str:
        if not logs:
            return "=== Logs ===\n\n(no logs captured)"
        return f"=== Logs ===\n\n{logs}"
