"""
optimization_engine.py

Responsible for:
- Translating static analysis hotspots into optimization strategies
- Building an optimization plan for codegen.py
- Coordinating Cython/Numba/preallocation/layout strategies
- Producing structured results for GUI and backend pipeline
"""

from dataclasses import dataclass
from typing import List, Dict

from .static_analysis import Hotspot, AnalysisResult


# ============================================================
# Data structures
# ============================================================

@dataclass
class OptimizationStrategy:
    """Represents a single optimization strategy."""
    name: str
    enabled: bool
    description: str


@dataclass
class OptimizationPlan:
    """
    Full optimization plan produced by the engine.
    This is consumed by codegen.py.
    """
    strategies: List[OptimizationStrategy]
    hotspots: List[Hotspot]
    cython_required: bool
    numba_required: bool
    preallocation_required: bool
    layout_opt_required: bool
    notes: List[str]


# ============================================================
# Optimization Engine
# ============================================================

class OptimizationEngine:
    """
    Converts static analysis results + user-selected strategies
    into a structured optimization plan.
    """

    def __init__(self):
        # Default strategies (GUI toggles will override these)
        self.available_strategies = {
            "cython_memoryviews": OptimizationStrategy(
                name="cython_memoryviews",
                enabled=False,
                description="Use Cython memoryviews and arena allocator."
            ),
            "numba_jit": OptimizationStrategy(
                name="numba_jit",
                enabled=False,
                description="Apply Numba JIT to hotspot functions."
            ),
            "preallocate_buffers": OptimizationStrategy(
                name="preallocate_buffers",
                enabled=False,
                description="Move allocations outside loops and preallocate buffers."
            ),
            "optimize_layout": OptimizationStrategy(
                name="optimize_layout",
                enabled=False,
                description="Ensure contiguous memory layout and SoA transformations."
            ),
        }

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def build_plan(
        self,
        analysis: AnalysisResult,
        user_strategy_selection: Dict[str, bool]
    ) -> OptimizationPlan:
        """
        Build a full optimization plan based on:
        - static analysis hotspots
        - user-selected strategies (from GUI)
        """

        # Update strategy states based on GUI selection
        for key, enabled in user_strategy_selection.items():
            if key in self.available_strategies:
                self.available_strategies[key].enabled = enabled

        # Determine required transformations
        cython_required = self._requires_cython(analysis)
        numba_required = self._requires_numba(analysis)
        prealloc_required = self._requires_preallocation(analysis)
        layout_required = self._requires_layout_opt(analysis)

        notes = self._generate_notes(
            cython_required,
            numba_required,
            prealloc_required,
            layout_required,
            analysis.hotspots
        )

        return OptimizationPlan(
            strategies=list(self.available_strategies.values()),
            hotspots=analysis.hotspots,
            cython_required=cython_required,
            numba_required=numba_required,
            preallocation_required=prealloc_required,
            layout_opt_required=layout_required,
            notes=notes
        )

    # --------------------------------------------------------
    # Strategy requirement detectors
    # --------------------------------------------------------

    def _requires_cython(self, analysis: AnalysisResult) -> bool:
        """
        Cython is required if temporary arrays or repeated allocations appear.
        """
        for h in analysis.hotspots:
            if h.type in ("temporary_array", "repeated_allocation"):
                return True
        return False

    def _requires_numba(self, analysis: AnalysisResult) -> bool:
        """
        Numba is required for nested loops or heavy numeric kernels.
        """
        for h in analysis.hotspots:
            if h.type == "nested_loop":
                return True
        return False

    def _requires_preallocation(self, analysis: AnalysisResult) -> bool:
        """
        Preallocation is required if large or repeated allocations appear.
        """
        for h in analysis.hotspots:
            if h.type in ("large_allocation", "repeated_allocation"):
                return True
        return False

    def _requires_layout_opt(self, analysis: AnalysisResult) -> bool:
        """
        Layout optimization is required if large arrays or slicing patterns appear.
        """
        for h in analysis.hotspots:
            if h.type == "large_allocation":
                return True
        return False

    # --------------------------------------------------------
    # Notes for GUI
    # --------------------------------------------------------

    def _generate_notes(
        self,
        cython_required: bool,
        numba_required: bool,
        prealloc_required: bool,
        layout_required: bool,
        hotspots: List[Hotspot]
    ) -> List[str]:
        """Generate human-readable notes for the GUI."""

        notes = []

        if cython_required:
            notes.append("Cython memoryviews recommended due to temporary or repeated allocations.")

        if numba_required:
            notes.append("Numba JIT recommended due to nested loops.")

        if prealloc_required:
            notes.append("Preallocation recommended due to large or repeated allocations.")

        if layout_required:
            notes.append("Memory layout optimization recommended for large arrays.")

        # Add hotspot-specific notes
        for h in hotspots:
            notes.append(f"Hotspot at line {h.lineno}: {h.description}")

        return notes
