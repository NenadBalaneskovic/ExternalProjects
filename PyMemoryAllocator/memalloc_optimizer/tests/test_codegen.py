"""
Unit tests for memalloc_core.codegen.CodeGenerator

Covers:
- Python code generation
- Cython code generation
- Output directory creation
- Notes correctness
- Structural correctness of CodegenResult
- Behavior with empty optimization plans
"""

import ast
import tempfile
from pathlib import Path
import pytest

from memalloc_core.codegen import CodeGenerator, CodegenResult
from memalloc_core.optimization_engine import OptimizationPlan, OptimizationStrategy


# ============================================================
# Helpers
# ============================================================

def make_plan(strategies):
    """Create an OptimizationPlan with given strategies."""
    return OptimizationPlan(
        strategies=strategies,
        hotspots=[],
        notes=["note1", "note2"],
        cython_required=any(s.enabled and s.name == "cython_memoryviews" for s in strategies),
        numba_required=any(s.enabled and s.name == "numba_jit" for s in strategies),
        preallocation_required=any(s.enabled and s.name == "preallocate_buffers" for s in strategies),
        layout_opt_required=any(s.enabled and s.name == "optimize_layout" for s in strategies),
    )


def make_ast():
    """Simple AST for testing."""
    code = """
def f():
    x = [i for i in range(100)]
    return sum(x)
"""
    return ast.parse(code)


def make_strategy(name, enabled=True):
    return OptimizationStrategy(
        name=name,
        enabled=enabled,
        description=f"desc for {name}",
    )


# ============================================================
# Tests
# ============================================================

def test_python_codegen(tmp_path):
    output_dir = tmp_path / "generated"
    cg = CodeGenerator(output_dir)

    plan = make_plan([make_strategy("cython_memoryviews", enabled=False)])
    tree = make_ast()

    result = cg.generate(plan, tree)

    assert isinstance(result, CodegenResult)
    assert result.optimized_python is not None
    assert result.optimized_python.exists()
    assert "note1" in result.notes


def test_cython_codegen(tmp_path):
    output_dir = tmp_path / "generated"
    cg = CodeGenerator(output_dir)

    plan = make_plan([make_strategy("cython_memoryviews", enabled=True)])
    tree = make_ast()

    result = cg.generate(plan, tree)

    assert isinstance(result, CodegenResult)
    assert result.optimized_cython is not None
    assert result.optimized_cython.exists()
    assert result.optimized_python.exists()  # Python always generated


def test_output_directory_created(tmp_path):
    output_dir = tmp_path / "generated"
    cg = CodeGenerator(output_dir)

    plan = make_plan([])
    tree = make_ast()

    cg.generate(plan, tree)

    assert output_dir.exists()
    assert any(output_dir.iterdir())


def test_empty_plan_generates_python(tmp_path):
    output_dir = tmp_path / "generated"
    cg = CodeGenerator(output_dir)

    plan = make_plan([])
    tree = make_ast()

    result = cg.generate(plan, tree)

    assert result.optimized_python.exists()
    assert result.optimized_cython is None  # no cython_required


def test_disabled_strategies_do_not_generate_cython(tmp_path):
    output_dir = tmp_path / "generated"
    cg = CodeGenerator(output_dir)

    plan = make_plan([
        make_strategy("cython_memoryviews", enabled=False),
        make_strategy("numba_jit", enabled=False),
    ])
    tree = make_ast()

    result = cg.generate(plan, tree)

    assert result.optimized_cython is None
    assert result.optimized_python.exists()


def test_codegen_result_structure(tmp_path):
    output_dir = tmp_path / "generated"
    cg = CodeGenerator(output_dir)

    plan = make_plan([make_strategy("preallocate_buffers", enabled=True)])
    tree = make_ast()

    result = cg.generate(plan, tree)

    assert hasattr(result, "optimized_python")
    assert hasattr(result, "optimized_cython")
    assert hasattr(result, "notes")
    assert isinstance(result.notes, list)
