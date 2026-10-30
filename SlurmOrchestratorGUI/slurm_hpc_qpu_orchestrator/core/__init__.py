"""
core package for the Slurm HPC–QPU Workflow Orchestrator.

This package provides:
- Safe AST parsing of uploaded Python workflow files
- Workflow classification (classical, quantum, hybrid)
- Slurm template generation with placeholder substitution
- Access to template library and configuration defaults

The core package NEVER executes user code. All analysis is static.
"""

# Public API re‑exports
from .ast_parser import ASTParser
from .workflow_classifier import WorkflowClassifier, WorkflowType
from .slurm_template_engine import SlurmTemplateEngine

# Convenience imports for template paths
from .template_library import (
    CLASSICAL_TEMPLATE_PATH,
    QUANTUM_TEMPLATE_PATH,
    HYBRID_TEMPLATE_PATH,
)

# Versioning (optional but recommended)
__version__ = "0.1.0"

__all__ = [
    "ASTParser",
    "WorkflowClassifier",
    "WorkflowType",
    "SlurmTemplateEngine",
    "CLASSICAL_TEMPLATE_PATH",
    "QUANTUM_TEMPLATE_PATH",
    "HYBRID_TEMPLATE_PATH",
]