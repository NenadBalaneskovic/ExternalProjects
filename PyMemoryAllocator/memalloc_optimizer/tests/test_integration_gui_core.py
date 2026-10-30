"""
Integration tests for GUI <-> Core pipeline using MemAllocController.

Covers:
- Script loading
- Static analysis
- Optimization plan generation
- Code generation
- Baseline execution
- Optimized execution
- Plot generation
- Metrics storage

These tests validate the full end-to-end pipeline without invoking the GUI event loop.
"""

import tempfile
from pathlib import Path
import pytest

from memalloc_gui.controllers import MemAllocController


# ============================================================
# Helpers
# ============================================================

def write_temp_script(code: str) -> Path:
    """Create a temporary Python script file."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    path = Path(tmp.name)
    path.write_text(code)
    return path


def make_controller(tmpdir: Path) -> MemAllocController:
    """Create a controller with isolated directories."""
    data_dir = tmpdir / "data"
    output_dir = tmpdir / "generated"
    plots_dir = tmpdir / "plots"
    db_path = tmpdir / "metrics.duckdb"

    return MemAllocController(
        data_dir=data_dir,
        output_dir=output_dir,
        db_path=db_path,
        plots_dir=plots_dir,
    )


# ============================================================
# Integration Tests
# ============================================================

def test_full_pipeline_simple(tmp_path):
    controller = make_controller(tmp_path)

    # --------------------------------------------------------
    # 1. Script loading
    # --------------------------------------------------------
    script = write_temp_script("""
import numpy as np

def f():
    x = np.zeros(1000)
    return x.sum()

if __name__ == "__main__":
    print(f())
""")

    load_result = controller.load_script(script)
    assert "hash" in load_result
    assert load_result["cached"] in (True, False)

    # --------------------------------------------------------
    # 2. Static analysis
    # --------------------------------------------------------
    analysis = controller.run_analysis()
    assert "hotspots" in analysis
    assert isinstance(analysis["tips"], list)

    # --------------------------------------------------------
    # 3. Optimization plan
    # --------------------------------------------------------
    user_selection = {
        "cython_memoryviews": True,
        "numba_jit": False,
        "preallocate_buffers": True,
        "optimize_layout": False,
    }

    plan = controller.build_plan(user_selection)
    assert "strategies" in plan
    assert len(plan["strategies"]) > 0

    # --------------------------------------------------------
    # 4. Code generation
    # --------------------------------------------------------
    codegen = controller.generate_code()
    assert codegen["python_generated"] is True
    # Cython may or may not be generated depending on plan
    assert "notes" in codegen

    # --------------------------------------------------------
    # 5. Baseline execution
    # --------------------------------------------------------
    baseline = controller.run_baseline()
    assert baseline["success"] is True
    assert baseline["runtime"] > 0
    assert baseline["memory"] >= 0

    # --------------------------------------------------------
    # 6. Optimized execution
    # --------------------------------------------------------
    optimized = controller.run_optimized()
    # Optimized may fail if Cython not generated; still test structure
    assert "success" in optimized
    assert "runtime" in optimized
    assert "memory" in optimized

    # --------------------------------------------------------
    # 7. Metrics storage
    # --------------------------------------------------------
    controller.store_metric(
        script_hash=load_result["hash"],
        runtime_seconds=baseline["runtime"],
        peak_memory_mb=baseline["memory"],
        speedup=1.0,
        strategy_summary="test summary",
    )

    metrics = controller.get_metrics()
    assert len(metrics["metrics"]) >= 1

    # --------------------------------------------------------
    # 8. Plot generation
    # --------------------------------------------------------
    plots = controller.generate_plots()
    assert "memory_plot" in plots
    # Plot may be None if no metrics yet, but structure must exist


def test_pipeline_script_error(tmp_path):
    controller = make_controller(tmp_path)

    script = write_temp_script("""
raise RuntimeError("boom")
""")

    controller.load_script(script)

    baseline = controller.run_baseline()
    assert baseline["success"] is False
    assert "boom" in baseline["stderr"].lower() or "boom" in baseline.get("error", "").lower()
