"""
view_models.py

GUI-facing data models for the MemAlloc Optimizer.
These models wrap backend results into clean, typed structures
that the GUI can render without touching backend internals.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict


# ============================================================
# Script Loading View Model
# ============================================================

@dataclass(frozen=True)
class ScriptLoadVM:
    path: str
    hash: str
    imports: List[str]
    entry_points: List[str]
    cached: bool


# ============================================================
# Static Analysis View Model
# ============================================================

@dataclass(frozen=True)
class HotspotVM:
    line: int
    type: str
    description: str


@dataclass(frozen=True)
class AnalysisVM:
    hotspots: List[HotspotVM]
    tips: List[str]


# ============================================================
# Optimization Plan View Model
# ============================================================

@dataclass(frozen=True)
class StrategyVM:
    name: str
    enabled: bool
    description: str


@dataclass(frozen=True)
class OptimizationPlanVM:
    strategies: List[StrategyVM]
    notes: List[str]


# ============================================================
# Code Generation View Model
# ============================================================

@dataclass(frozen=True)
class CodegenVM:
    python_generated: bool
    cython_generated: bool
    notes: List[str]


# ============================================================
# Execution View Model
# ============================================================

@dataclass(frozen=True)
class ExecutionVM:
    success: bool
    runtime: float
    memory: float
    stdout: str
    stderr: str
    error: Optional[str] = None


# ============================================================
# Metrics View Model
# ============================================================

@dataclass(frozen=True)
class MetricVM:
    timestamp: str
    script_hash: str
    runtime_seconds: float
    peak_memory_mb: float
    speedup: float
    strategy_summary: str


@dataclass(frozen=True)
class MetricsVM:
    metrics: List[MetricVM]


# ============================================================
# Plots View Model
# ============================================================

@dataclass(frozen=True)
class PlotsVM:
    memory_plot: Optional[str]
    runtime_plot: Optional[str]
    speedup_plot: Optional[str]
