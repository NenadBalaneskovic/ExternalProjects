# simulation/fem_solver.py
"""
Finite Element Method (FEM) solver for Sensor Stress Analyzer.
Performs stress and heat distribution analysis on polygonal rod structures.
"""

import numpy as np
from simulation.mesh_generator import generate_mesh
from simulation.solver_utils import solve_stress, solve_heat


def run_fem_analysis(n: int, force: float, heat: float) -> dict:
    """
    Perform FEM analysis on an n-gon rod structure.
    Returns stress and heat distributions along with maxima for reporting.
    """

    # --- Mesh generation ---
    mesh = generate_mesh(n)

    # --- FEM stress solver ---
    stress_map = solve_stress(mesh, force)
    stress_max = float(np.max(stress_map))

    # --- FEM heat solver ---
    heat_map = solve_heat(mesh, heat)
    heat_max = float(np.max(heat_map))

    # --- Results dictionary ---
    results = {
        "mode": "FEM Analysis",
        "n": n,
        "force": force,
        "heat": heat,
        "mesh": mesh.tolist() if hasattr(mesh, "tolist") else mesh,
        "stress_map": stress_map.tolist() if hasattr(stress_map, "tolist") else stress_map,
        "heat_map": heat_map.tolist() if hasattr(heat_map, "tolist") else heat_map,
        "stress_max": stress_max,
        "heat_max": heat_max,
    }

    return results
