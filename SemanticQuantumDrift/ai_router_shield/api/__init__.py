# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  api/__init__.py
ROLE:    Package initializer elevating the network transport bridge and 
         the non-linear web thread orchestrator to the unified controller namespace.
================================================================================
"""

# Explicit promotion of transport subsystems to ease modular cross-layer controller imports
from api.web_bridge import BoundedWebSessionBridge
from api.thread_manager import ThreadContextOrchestrator

# Define the explicit public API export layout of the api infrastructure package
__all__ = [
    "BoundedWebSessionBridge",
    "ThreadContextOrchestrator"
]
