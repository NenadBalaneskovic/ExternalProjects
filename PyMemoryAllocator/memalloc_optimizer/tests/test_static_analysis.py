"""
Unit tests for memalloc_core.static_analysis.StaticAnalyzer

Covers:
- Hotspot detection
- Memory tip generation
- AST parsing correctness
- Edge cases (no hotspots, multiple hotspots)
"""

import ast
import pytest

from memalloc_core.static_analysis import StaticAnalyzer, Hotspot, AnalysisResult


# ============================================================
# Helpers
# ============================================================

def analyze(code: str) -> AnalysisResult:
    """Parse code into AST and run static analysis."""
    tree = ast.parse(code)
    analyzer = StaticAnalyzer()
    return analyzer.analyze(tree)


# ============================================================
# Tests
# ============================================================

def test_detect_temporary_array():
    code = """
import numpy as np

def f():
    x = np.zeros(1000)   # temporary array
    return x.sum()
"""
    result = analyze(code)

    assert len(result.hotspots) == 1
    h = result.hotspots[0]

    assert h.type == "temporary_array"
    assert "temporary" in h.description.lower()


def test_detect_repeated_allocation():
    code = """
import numpy as np

def f():
    for i in range(10):
        x = np.zeros(1000)   # repeated allocation
"""
    result = analyze(code)

    assert len(result.hotspots) == 1
    assert result.hotspots[0].type == "repeated_allocation"


def test_detect_nested_loop():
    code = """
def f():
    for i in range(10):
        for j in range(10):
            pass
"""
    result = analyze(code)

    assert len(result.hotspots) == 1
    assert result.hotspots[0].type == "nested_loop"


def test_detect_large_allocation():
    code = """
import numpy as np

def f():
    x = np.zeros(10_000_000)   # large allocation
"""
    result = analyze(code)

    assert len(result.hotspots) == 1
    assert result.hotspots[0].type == "large_allocation"


def test_multiple_hotspots():
    code = """
import numpy as np

def f():
    x = np.zeros(1000)        # temporary
    for i in range(10):
        y = np.zeros(500)     # repeated
    for i in range(5):
        for j in range(5):    # nested
            pass
"""
    result = analyze(code)

    types = {h.type for h in result.hotspots}

    assert "temporary_array" in types
    assert "repeated_allocation" in types
    assert "nested_loop" in types
    assert len(result.hotspots) == 3


def test_memory_tips_present():
    code = """
import numpy as np

def f():
    x = np.zeros(1000)
"""
    result = analyze(code)

    assert len(result.memory_tips) > 0
    assert any("contiguous" in tip.lower() for tip in result.memory_tips)


def test_no_hotspots():
    code = """
def f():
    return 42
"""
    result = analyze(code)

    assert result.hotspots == []
    assert isinstance(result.memory_tips, list)
