# simulation/solver_utils.py
"""
Solver utilities for Sensor Stress Analyzer.
Provides simplified FEM stress and heat solvers, plus acceleration, benchmarking, and validation helpers.
"""

import time
import numpy as np


def solve_stress(mesh: dict, force: float) -> np.ndarray:
    """
    Simplified FEM stress solver.
    Args:
        mesh (dict): Mesh data including vertices and edges.
        force (float): Applied force [N].
    Returns:
        np.ndarray: Stress values per vertex.
    """
    vertices = np.array(mesh["vertices"], dtype=float)
    n = len(vertices)

    # Stress proportional to force and vertex angle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    stress_map = force * (1 + 0.1 * np.sin(angles))

    return stress_map.astype(float)


def solve_heat(mesh: dict, heat: float) -> np.ndarray:
    """
    Simplified FEM heat solver.
    Args:
        mesh (dict): Mesh data including vertices and edges.
        heat (float): Applied heat [°C].
    Returns:
        np.ndarray: Heat values per vertex.
    """
    vertices = np.array(mesh["vertices"], dtype=float)
    n = len(vertices)

    # Heat proportional to applied temperature and vertex angle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    heat_map = heat * (1 + 0.05 * np.cos(angles))

    return heat_map.astype(float)


def accelerate_solver(data):
    """
    Placeholder for solver acceleration using Cython/CPython.
    In a real implementation, this would call optimized routines
    compiled from Cython or C extensions to speed up FEM calculations.

    Args:
        data (dict or list): Input data for solver.

    Returns:
        dict or list: Processed data (currently unchanged).
    """
    # Simulate acceleration by returning data directly
    return data


def benchmark_solver(func, *args, **kwargs):
    """
    Benchmark the execution time of a solver function.

    Args:
        func (callable): Solver function to benchmark.
        *args: Positional arguments for the solver.
        **kwargs: Keyword arguments for the solver.

    Returns:
        tuple: (results, elapsed_time)
    """
    start = time.time()
    results = func(*args, **kwargs)
    elapsed = time.time() - start
    return results, elapsed


def validate_results(results):
    """
    Validate solver results for consistency and completeness.

    Args:
        results (dict): Results dictionary from solver.

    Returns:
        bool: True if results are valid, False otherwise.
    """
    if not isinstance(results, dict):
        return False
    required_keys = ["mode", "stress_map", "heat_map", "stress_max", "heat_max"]
    return all(key in results for key in required_keys)
