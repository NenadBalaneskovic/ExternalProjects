# tests/test_rod_analysis.py

import pytest
from simulation.rod_analysis import run_rod_analysis


def test_rod_analysis_basic():
    results = run_rod_analysis(6, 50, 30)
    assert isinstance(results, dict)
    assert "max_deflection" in results
    assert "critical_corner" in results
    assert len(results["deflections"]) == 6


def test_rod_analysis_range():
    results = run_rod_analysis(12, 10, 5)
    assert min(results["deflections"]) <= max(results["deflections"])
    assert results["n"] == 12