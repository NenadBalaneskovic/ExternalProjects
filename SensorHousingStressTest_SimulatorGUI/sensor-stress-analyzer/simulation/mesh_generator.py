# simulation/mesh_generator.py
"""
Mesh generator for Sensor Stress Analyzer.
Creates polygonal meshes for FEM analysis of rod structures.
"""

import math
import numpy as np


def generate_polygon_vertices(n: int, radius: float = 1.0) -> np.ndarray:
    """
    Generate vertices of a regular n-gon polygon.
    Args:
        n (int): Number of polygon corners (3–21).
        radius (float): Radius of circumscribed circle.
    Returns:
        np.ndarray: Array of shape (n, 2) with (x, y) coordinates of vertices.
    """
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    vertices = np.column_stack((x, y))
    return vertices.astype(float)


def generate_polygon_edges(vertices: np.ndarray) -> np.ndarray:
    """
    Generate edges connecting consecutive vertices of the polygon.
    Args:
        vertices (np.ndarray): Array of (x, y) coordinates.
    Returns:
        np.ndarray: Array of shape (n, 2) with vertex index pairs.
    """
    n = len(vertices)
    edges = [(i, (i + 1) % n) for i in range(n)]  # wrap around to close polygon
    return np.array(edges, dtype=int)


def generate_polygon_mesh(n: int, radius: float = 1.0) -> dict:
    """
    Generate a polygon mesh structure for FEM analysis.
    Args:
        n (int): Number of polygon corners.
        radius (float): Radius of circumscribed circle.
    Returns:
        dict: Mesh data including vertices and edges (NumPy arrays).
    """
    vertices = generate_polygon_vertices(n, radius)
    edges = generate_polygon_edges(vertices)

    mesh = {
        "n": n,
        "vertices": vertices,
        "edges": edges,
    }
    return mesh


# --- Alias for backward compatibility ---
# Allows fem_solver.py to import generate_mesh without error
generate_mesh = generate_polygon_mesh
