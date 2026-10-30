"""
runtime_profiler.py

Responsible for:
- Measuring runtime performance of Python scripts/functions
- Tracking memory allocations via tracemalloc
- Measuring process-level memory usage via psutil
- Producing structured profiler results for GUI and metrics store
"""

import tracemalloc
import psutil
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable


# ============================================================
# Data structures
# ============================================================

@dataclass
class ProfileResult:
    """Structured result of profiling a script or function."""
    runtime_seconds: float
    peak_memory_mb: float
    allocations: int
    snapshot_top: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


# ============================================================
# Runtime Profiler
# ============================================================

class RuntimeProfiler:
    """
    Profiles Python scripts or functions using:
    - tracemalloc for allocation tracking
    - psutil for process-level memory usage
    - perf_counter for runtime measurement
    """

    def __init__(self):
        pass

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def profile_script(self, script_path: Path) -> ProfileResult:
        """
        Run a Python script in a separate process and profile:
        - runtime
        - peak memory usage
        - allocation statistics (not available for subprocess)
        """

        try:
            start_time = time.perf_counter()

            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            peak_memory = 0.0
            ps_proc = psutil.Process(process.pid)

            # Monitor memory usage
            while process.poll() is None:
                try:
                    mem = ps_proc.memory_info().rss / (1024 ** 2)
                    peak_memory = max(peak_memory, mem)
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.01)

            stdout, stderr = process.communicate()
            end_time = time.perf_counter()

            success = process.returncode == 0

            return ProfileResult(
                runtime_seconds=end_time - start_time,
                peak_memory_mb=peak_memory,
                allocations=0,          # subprocess cannot return tracemalloc data
                snapshot_top={},        # no snapshot available
                success=success,
                error_message=None if success else stderr
            )

        except Exception as e:
            return ProfileResult(
                runtime_seconds=0.0,
                peak_memory_mb=0.0,
                allocations=0,
                snapshot_top={},
                success=False,
                error_message=str(e)
            )

    # --------------------------------------------------------
    # Function-level profiling
    # --------------------------------------------------------

    def profile_function(self, func: Callable, *args, **kwargs) -> ProfileResult:
        """
        Profile a Python function directly inside the current process.
        This allows:
        - tracemalloc snapshots
        - allocation counts
        - top allocation lines
        """

        try:
            tracemalloc.start()

            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()

            # Allocation statistics
            current, peak = tracemalloc.get_traced_memory()
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")

            tracemalloc.stop()

            peak_mb = peak / (1024 ** 2)

            # Extract top allocation lines
            top = {}
            for stat in top_stats[:10]:
                tb = stat.traceback[0]
                key = f"{tb.filename}:{tb.lineno}"
                top[key] = stat.size

            return ProfileResult(
                runtime_seconds=end_time - start_time,
                peak_memory_mb=peak_mb,
                allocations=len(snapshot.traces),
                snapshot_top=top,
                success=True
            )

        except Exception as e:
            tracemalloc.stop()
            return ProfileResult(
                runtime_seconds=0.0,
                peak_memory_mb=0.0,
                allocations=0,
                snapshot_top={},
                success=False,
                error_message=str(e)
            )
