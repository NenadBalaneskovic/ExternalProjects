"""
memalloc_gui package

This package contains the full GUI layer for the MemAlloc Optimizer.
It provides:
- The main application entry point (app.py)
- The controller layer (controllers.py)
- The layout system (layout.py)
- The theming system (theming.py)
- GUI-facing view models (view_models.py)

The __init__.py file exposes the public API for launching the GUI.
"""

from .app import MemAllocApp, main
from .controllers import MemAllocController
from .layout import build_main_layout
from .theming import ThemeLoader, apply_theme
from .view_models import (
    ScriptLoadVM,
    AnalysisVM,
    OptimizationPlanVM,
    CodegenVM,
    ExecutionVM,
    MetricsVM,
    PlotsVM,
)

__all__ = [
    "MemAllocApp",
    "main",
    "MemAllocController",
    "build_main_layout",
    "ThemeLoader",
    "apply_theme",
    "ScriptLoadVM",
    "AnalysisVM",
    "OptimizationPlanVM",
    "CodegenVM",
    "ExecutionVM",
    "MetricsVM",
    "PlotsVM",
]
