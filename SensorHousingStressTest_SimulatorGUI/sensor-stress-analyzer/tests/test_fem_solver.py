# tests/test_fem_solver.py

import pytest
from simulation.fem_solver import run_fem_analysis


def test_fem_solver_basic():
    results = run_fem_analysis(6, 50, 30)
    assert isinstance(results, dict)
    assert "stress_map" in results
    assert "heat_map" in results
    assert results["max_stress"] == max(results["stress_map"])
    assert results["max_heat"] == max(results["heat_map"])


def test_fem_solver_length():
    results = run_fem_analysis(8, 20, 10)
    assert len(results["stress_map"]) == 8
    assert len(results["heat_map"]) == 8