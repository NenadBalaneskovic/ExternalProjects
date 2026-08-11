"""
Unit tests for memalloc_core.runtime_profiler.RuntimeProfiler

Covers:
- Function-level profiling (tracemalloc)
- Script-level profiling (subprocess + psutil)
- Error handling
- Structural correctness of ProfileResult
"""

import time
import tempfile
from pathlib import Path
import pytest

from memalloc_core.runtime_profiler import RuntimeProfiler, ProfileResult


# ============================================================
# Helpers
# ============================================================

def write_temp_script(code: str) -> Path:
    """Create a temporary Python script file."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    path = Path(tmp.name)
    path.write_text(code)
    return path


# ============================================================
# Function-level profiling tests
# ============================================================

def test_profile_function_simple():
    profiler = RuntimeProfiler()

    def test_func():
        lst = [i for i in range(1000)]
        return sum(lst)

    result = profiler.profile_function(test_func)

    assert isinstance(result, ProfileResult)
    assert result.success is True
    assert result.runtime_seconds > 0
    assert result.peak_memory_mb > 0
    assert result.allocations > 0
    assert isinstance(result.snapshot_top, dict)


def test_profile_function_exception():
    profiler = RuntimeProfiler()

    def bad_func():
        raise ValueError("boom")

    result = profiler.profile_function(bad_func)

    assert result.success is False
    assert "boom" in result.error_message.lower()
    assert result.runtime_seconds == 0.0
    assert result.peak_memory_mb == 0.0
    assert result.allocations == 0


# ============================================================
# Script-level profiling tests
# ============================================================

def test_profile_script_simple():
    profiler = RuntimeProfiler()

    script = write_temp_script("""
import time
x = [i for i in range(10000)]
time.sleep(0.05)
print("OK")
""")

    result = profiler.profile_script(script)

    assert isinstance(result, ProfileResult)
    assert result.success is True
    assert result.runtime_seconds > 0
    assert result.peak_memory_mb > 0
    assert result.stdout.strip() == "OK"
    assert result.stderr.strip() == ""


def test_profile_script_error():
    profiler = RuntimeProfiler()

    script = write_temp_script("""
raise RuntimeError("script failed")
""")

    result = profiler.profile_script(script)

    assert result.success is False
    assert "script failed" in result.error_message.lower()
    assert result.runtime_seconds > 0
    assert result.peak_memory_mb >= 0


def test_profile_script_large_memory():
    profiler = RuntimeProfiler()

    script = write_temp_script("""
# allocate ~10 MB
import numpy as np
x = np.zeros(1_250_000, dtype=np.float64)
print("done")
""")

    result = profiler.profile_script(script)

    assert result.success is True
    assert result.peak_memory_mb > 5  # MB
    assert "done" in result.stdout


# ============================================================
# Structural correctness
# ============================================================

def test_profile_result_structure():
    profiler = RuntimeProfiler()

    def f():
        return 123

    result = profiler.profile_function(f)

    assert hasattr(result, "runtime_seconds")
    assert hasattr(result, "peak_memory_mb")
    assert hasattr(result, "allocations")
    assert hasattr(result, "snapshot_top")
    assert hasattr(result, "success")
    assert hasattr(result, "error_message")
