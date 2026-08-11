"""
execution_sandbox.py

Responsible for:
- Safely executing baseline and optimized Python scripts
- Executing compiled Cython modules
- Isolating execution in subprocesses
- Capturing stdout, stderr, exit codes
- Returning structured results for GUI and metrics store
"""

import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import os

import psutil


# ============================================================
# Data structures
# ============================================================

@dataclass
class ExecutionResult:
    """Structured result of sandbox execution."""
    success: bool
    runtime_seconds: float
    peak_memory_mb: float
    stdout: str
    stderr: str
    error_message: Optional[str] = None


# ============================================================
# Execution Sandbox
# ============================================================

class ExecutionSandbox:
    """
    Safely executes Python scripts or optimized modules in isolated subprocesses.
    Tracks:
    - runtime
    - peak memory usage
    - stdout/stderr
    """

    def __init__(self):
        pass

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def run_script(self, script_path: Path) -> ExecutionResult:
        """
        Execute a Python script in a subprocess.
        Used for baseline execution and optimized Python modules.
        """

        try:
            start_time = time.perf_counter()

            env = os.environ.copy()
            env["NUMBA_DISABLE_JIT"] = "1"   # ⭐ prevents JIT pipe blocking
            
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(script_path.parent),  # ⭐ ensures correct imports
                env=env                       # ⭐ ensures stable stdout
            )

            ps_proc = psutil.Process(process.pid)
            peak_memory = 0.0

            # Initial memory sample (critical for ultra-fast scripts)
            try:
                mem = ps_proc.memory_info().rss / (1024 ** 2)
                peak_memory = max(peak_memory, mem)
            except psutil.NoSuchProcess:
                pass

            # High‑resolution memory sampling (1 ms)
            while True:
                if process.poll() is not None:
                    # Final memory sample
                    try:
                        mem = ps_proc.memory_info().rss / (1024 ** 2)
                        peak_memory = max(peak_memory, mem)
                    except psutil.NoSuchProcess:
                        pass

                    # ⭐ Correct place to capture stdout/stderr
                    stdout, stderr = process.communicate()
                    break

                try:
                    mem = ps_proc.memory_info().rss / (1024 ** 2)
                    peak_memory = max(peak_memory, mem)
                except psutil.NoSuchProcess:
                    break

                time.sleep(0.001)

            end_time = time.perf_counter()
            success = process.returncode == 0

            return ExecutionResult(
                success=success,
                runtime_seconds=end_time - start_time,
                peak_memory_mb=peak_memory,
                stdout=stdout,
                stderr=stderr,
                error_message=None if success else stderr
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                runtime_seconds=0.0,
                peak_memory_mb=0.0,
                stdout="",
                stderr="",
                error_message=str(e)
            )

    # --------------------------------------------------------
    # Cython module execution
    # --------------------------------------------------------

    def run_cython_module(self, module_path: Path) -> ExecutionResult:
        """
        Execute a compiled Cython module.
        Assumes the module has a main() entry point.
        """

        try:
            start_time = time.perf_counter()

            module_name = module_path.stem

            env = os.environ.copy()
            env["NUMBA_DISABLE_JIT"] = "1"
            
            process = subprocess.Popen(
                [sys.executable, "-c", f"import {module_name}; {module_name}.main()"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(module_path.parent),
                env=env
            )

            ps_proc = psutil.Process(process.pid)
            peak_memory = 0.0

            # Initial memory sample
            try:
                mem = ps_proc.memory_info().rss / (1024 ** 2)
                peak_memory = max(peak_memory, mem)
            except psutil.NoSuchProcess:
                pass

            # High‑resolution memory sampling
            while True:
                if process.poll() is not None:
                    try:
                        mem = ps_proc.memory_info().rss / (1024 ** 2)
                        peak_memory = max(peak_memory, mem)
                    except psutil.NoSuchProcess:
                        pass

                    # ⭐ Correct stdout/stderr capture
                    stdout, stderr = process.communicate()
                    break

                try:
                    mem = ps_proc.memory_info().rss / (1024 ** 2)
                    peak_memory = max(peak_memory, mem)
                except psutil.NoSuchProcess:
                    break

                time.sleep(0.001)

            end_time = time.perf_counter()
            success = process.returncode == 0

            return ExecutionResult(
                success=success,
                runtime_seconds=end_time - start_time,
                peak_memory_mb=peak_memory,
                stdout=stdout,
                stderr=stderr,
                error_message=None if success else stderr
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                runtime_seconds=0.0,
                peak_memory_mb=0.0,
                stdout="",
                stderr="",
                error_message=str(e)
            )
