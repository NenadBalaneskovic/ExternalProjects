"""
static_analysis.py

Responsible for:
- AST-based detection of memory hotspots
- Identifying large allocations, nested loops, repeated allocations
- Producing structured hotspot descriptors for the optimizer and GUI
"""

from dataclasses import dataclass
from typing import List, Dict
import ast


# ============================================================
# Data structures
# ============================================================

@dataclass
class Hotspot:
    """Represents a memory-relevant hotspot in the script."""
    type: str                     # e.g., "nested_loop", "large_allocation"
    lineno: int                   # line number in the script
    description: str              # human-readable explanation
    details: Dict                 # additional structured info


@dataclass
class AnalysisResult:
    """Full static analysis result."""
    hotspots: List[Hotspot]
    memory_tips: List[str]


# ============================================================
# Static Analyzer
# ============================================================

class StaticAnalyzer:
    """
    Performs AST-based static analysis to detect memory hotspots.
    """

    def __init__(self):
        pass

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def analyze(self, ast_tree: ast.AST) -> AnalysisResult:
        hotspots = []

        hotspots.extend(self._detect_nested_loops(ast_tree))
        hotspots.extend(self._detect_large_allocations(ast_tree))
        hotspots.extend(self._detect_repeated_allocations(ast_tree))
        hotspots.extend(self._detect_temp_arrays(ast_tree))

        memory_tips = self._generate_memory_tips(hotspots)

        return AnalysisResult(
            hotspots=hotspots,
            memory_tips=memory_tips
        )

    # --------------------------------------------------------
    # Hotspot detectors
    # --------------------------------------------------------

    def _detect_nested_loops(self, tree: ast.AST) -> List[Hotspot]:
        """Detect nested loops (O(n^2) or worse)."""
        hotspots = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                inner_loops = [
                    n for n in ast.walk(node)
                    if isinstance(n, (ast.For, ast.While))
                ]
                if len(inner_loops) > 1:
                    hotspots.append(
                        Hotspot(
                            type="nested_loop",
                            lineno=node.lineno,
                            description="Nested loop detected (potential O(n^2) memory behavior).",
                            details={"depth": len(inner_loops)}
                        )
                    )
        return hotspots

    def _detect_large_allocations(self, tree: ast.AST) -> List[Hotspot]:
        """Detect large NumPy or list allocations."""
        hotspots = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_func_name(node)

                # NumPy allocations
                if func_name in (
                    "np.zeros", "np.ones", "np.random.rand",
                    "numpy.zeros", "numpy.ones", "numpy.random.rand"
                ):
                    hotspots.append(
                        Hotspot(
                            type="large_allocation",
                            lineno=node.lineno,
                            description=f"Large array allocation via {func_name}.",
                            details={"func": func_name}
                        )
                    )

                # Large range allocations
                if isinstance(node.func, ast.Name) and node.func.id == "range":
                    if node.args and isinstance(node.args[0], ast.Constant):
                        if node.args[0].value > 10000:
                            hotspots.append(
                                Hotspot(
                                    type="large_allocation",
                                    lineno=node.lineno,
                                    description="Large range allocation detected.",
                                    details={"size": node.args[0].value}
                                )
                            )
        return hotspots

    def _detect_repeated_allocations(self, tree: ast.AST) -> List[Hotspot]:
        """Detect repeated allocations inside loops."""
        hotspots = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        func_name = self._get_func_name(inner)
                        if func_name in ("np.zeros", "np.ones", "numpy.zeros", "numpy.ones"):
                            hotspots.append(
                                Hotspot(
                                    type="repeated_allocation",
                                    lineno=inner.lineno,
                                    description=f"Repeated allocation inside loop via {func_name}.",
                                    details={"func": func_name}
                                )
                            )
        return hotspots

    def _detect_temp_arrays(self, tree: ast.AST) -> List[Hotspot]:
        """Detect temporary arrays created inside loops."""
        hotspots = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.BinOp):
                    # e.g., temp = data[i] * 0.5
                    if isinstance(node.value.left, ast.Subscript):
                        hotspots.append(
                            Hotspot(
                                type="temporary_array",
                                lineno=node.lineno,
                                description="Temporary array created inside loop.",
                                details={"target": self._get_target_name(node)}
                            )
                        )
        return hotspots

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _get_func_name(self, node: ast.Call) -> str:
        """Extract function name from AST Call node."""
        if isinstance(node.func, ast.Attribute):
            # e.g. np.zeros → ("np", "zeros")
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
        elif isinstance(node.func, ast.Name):
            return node.func.id
        return "unknown"

    def _get_target_name(self, node: ast.Assign) -> str:
        """Extract variable name from assignment."""
        if node.targets and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        return "unknown"

    # --------------------------------------------------------
    # Memory Tips
    # --------------------------------------------------------

    def _generate_memory_tips(self, hotspots: List[Hotspot]) -> List[str]:
        """Generate human-readable memory tips based on hotspots."""
        tips = []

        for h in hotspots:
            if h.type == "nested_loop":
                tips.append("Consider Numba JIT or Cython for nested loops.")
            if h.type == "large_allocation":
                tips.append("Large arrays detected — consider preallocation or memoryviews.")
            if h.type == "repeated_allocation":
                tips.append("Repeated allocations inside loops — move allocations outside.")
            if h.type == "temporary_array":
                tips.append("Temporary arrays inside loops — consider using preallocated buffers.")

        return list(set(tips))
