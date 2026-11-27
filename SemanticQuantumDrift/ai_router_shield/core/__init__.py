# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  core/__init__.py
ROLE:    Package initializer elevating the mathematical engine, semantic router,
         and SQLite database persistence layers to the unified core namespace.
================================================================================
"""

# Explicit promotion of subsystems to ease clean modular imports across layers
from core.metrics import QuantumRGDriftAnalyzer, HeterogeneousDiagnosticEngine
from core.database import SessionPersistenceManager
from core.router import IntentPreRoutingEngine

# Define the explicit public API export layout of the core architecture package
__all__ = [
    "QuantumRGDriftAnalyzer",
    "HeterogeneousDiagnosticEngine",
    "SessionPersistenceManager",
    "IntentPreRoutingEngine"
]
