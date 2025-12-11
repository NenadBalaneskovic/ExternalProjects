# simulation/__init__.py
"""
Simulation package for Sensor Stress Analyzer.
Provides rod analysis, FEM analysis, mesh generation, and solver utilities.
"""

from .rod_analysis import run_rod_analysis
from .fem_solver import run_fem_analysis
from .mesh_generator import generate_polygon_mesh
from .solver_utils import solve_stress, solve_heat

__all__ = [
    "run_rod_analysis",
    "run_fem_analysis",
    "generate_polygon_mesh",
    "solve_stress",
    "solve_heat",
]
