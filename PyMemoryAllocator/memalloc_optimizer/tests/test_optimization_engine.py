"""
Unit tests for memalloc_core.optimization_engine.OptimizationEngine

Covers:
- Strategy enabling/disabling
- Automatic requirement detection
- Hotspot-driven plan generation
- Notes correctness
- Structural correctness of OptimizationPlan
"""

import pytest

from memalloc_core.optimization_engine import (
    OptimizationEngine,
    OptimizationPlan,
    OptimizationStrategy,
)
from memalloc_core.static_analysis import Hotspot, AnalysisResult


# ============================================================
# Helpers
# ============================================================

def make_hotspot(lineno: int, type_: str, desc: str) -> Hotspot:
    return Hotspot(
        lineno=lineno,
        type=type_,
        description=desc,
    )


def make_analysis_result(hotspots):
    return AnalysisResult(
        hotspots=hotspots,
        memory_tips=["tip1", "tip2"],
    )


# ============================================================
# Tests
# ============================================================

def test_strategy_enable_disable():
    engine = OptimizationEngine()

    analysis = make_analysis_result([])

    user_selection = {
        "cython_memoryviews": True,
        "numba_jit": False,
        "preallocate_buffers": True,
        "optimize_layout": False,
    }

    plan = engine.build_plan(analysis, user_selection)

    strategies = {s.name: s.enabled for s in plan.strategies}

    assert strategies["cython_memoryviews"] is True
    assert strategies["numba_jit"] is False
    assert strategies["preallocate_buffers"] is True
    assert strategies["optimize_layout"] is False


def test_requires_cython():
    engine = OptimizationEngine()

    hotspots = [
        make_hotspot(10, "temporary_array", "temp array"),
        make_hotspot(20, "nested_loop", "nested"),
    ]
    analysis = make_analysis_result(hotspots)

    plan = engine.build_plan(analysis, {})

    assert plan.cython_required is True
    assert plan.numba_required is True
    assert plan.preallocation_required is False
    assert plan.layout_opt_required is False


def test_requires_preallocation_and_layout():
    engine = OptimizationEngine()

    hotspots = [
        make_hotspot(5, "large_allocation", "big array"),
        make_hotspot(15, "repeated_allocation", "repeat alloc"),
    ]
    analysis = make_analysis_result(hotspots)

    plan = engine.build_plan(analysis, {})

    assert plan.preallocation_required is True
    assert plan.layout_opt_required is True
    assert plan.cython_required is True  # repeated_allocation triggers cython
    assert plan.numba_required is False


def test_notes_generation():
    engine = OptimizationEngine()

    hotspots = [
        make_hotspot(42, "nested_loop", "nested loop"),
        make_hotspot(100, "temporary_array", "temp array"),
    ]
    analysis = make_analysis_result(hotspots)

    plan = engine.build_plan(analysis, {})

    notes = plan.notes

    assert any("Cython" in n for n in notes)
    assert any("Numba" in n for n in notes)
    assert any("Hotspot at line 42" in n for n in notes)
    assert any("Hotspot at line 100" in n for n in notes)


def test_plan_structure():
    engine = OptimizationEngine()

    analysis = make_analysis_result([])

    plan = engine.build_plan(analysis, {})

    assert isinstance(plan, OptimizationPlan)
    assert isinstance(plan.strategies, list)
    assert isinstance(plan.hotspots, list)
    assert isinstance(plan.notes, list)

    for s in plan.strategies:
        assert isinstance(s, OptimizationStrategy)
        assert hasattr(s, "name")
        assert hasattr(s, "enabled")
        assert hasattr(s, "description")
