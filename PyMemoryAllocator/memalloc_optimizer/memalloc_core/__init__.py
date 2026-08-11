"""
memalloc_core package

This package contains the full backend pipeline for the MemAlloc Optimizer:
- Script loading & metadata extraction
- Static memory hotspot analysis
- Runtime profiling (time + memory)
- Optimization strategy planning
- Code generation (Python + Cython)
- Safe execution sandbox
- Persistent metrics storage (DuckDB)

The __init__.py file exposes the public API for GUI and external modules.
"""

from .script_manager import ScriptManager, ScriptMetadata, ScriptLoadResult
from .static_analysis import StaticAnalyzer, AnalysisResult, Hotspot
from .runtime_profiler import RuntimeProfiler, ProfileResult
from .optimization_engine import OptimizationEngine, OptimizationPlan, OptimizationStrategy
from .codegen import CodeGenerator, CodegenResult
from .execution_sandbox import ExecutionSandbox, ExecutionResult
from .metrics_store import MetricsStore, MetricRecord

__all__ = [
    "ScriptManager",
    "ScriptMetadata",
    "ScriptLoadResult",
    "StaticAnalyzer",
    "AnalysisResult",
    "Hotspot",
    "RuntimeProfiler",
    "ProfileResult",
    "OptimizationEngine",
    "OptimizationPlan",
    "OptimizationStrategy",
    "CodeGenerator",
    "CodegenResult",
    "ExecutionSandbox",
    "ExecutionResult",
    "MetricsStore",
    "MetricRecord",
]
