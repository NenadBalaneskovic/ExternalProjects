"""
WorkflowClassifier
------------------
Classifies an uploaded Python workflow into one of three categories:

    - CLASSICAL: No quantum imports, no QPU calls.
    - QUANTUM: Quantum imports + QPU calls, but no classical optimizer loop.
    - HYBRID: Quantum imports + QPU calls + classical loop (optimizer pattern).

This module NEVER executes user code. Classification is purely static and
based on ASTParser output.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List

from .ast_parser import ParsedWorkflow


# ---------------------------------------------------------------------------
# Workflow Types
# ---------------------------------------------------------------------------

class WorkflowType(Enum):
    CLASSICAL = auto()
    QUANTUM = auto()
    HYBRID = auto()


# ---------------------------------------------------------------------------
# Classification Result
# ---------------------------------------------------------------------------

@dataclass
class WorkflowClassification:
    workflow_type: WorkflowType
    quantum_imports: List[str]
    quantum_calls: List[str]
    classical_imports: List[str]
    has_loops: bool


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class WorkflowClassifier:
    """
    Determines whether a workflow is classical, quantum, or hybrid.
    """

    # Strict quantum import prefixes
    QUANTUM_IMPORT_PREFIXES = [
        "qiskit",
        "qiskit_ibm_runtime",
        "qiskit_serverless",
        "qiskit_aer",
        "qiskit.providers",
        "braket",
        "cirq",
        "pennylane",
        "qutip",
        "azure.quantum",
    ]

    # Strict quantum call prefixes (NO bare "run")
    QUANTUM_CALL_PREFIXES = [
        "Sampler",
        "Estimator",
        "Session",
        "QuantumCircuit",
        "execute",
    ]

    # Classical heavy imports
    CLASSICAL_IMPORT_PREFIXES = [
        "numpy",
        "scipy",
        "torch",
        "tensorflow",
        "jax",
    ]

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def classify(self, parsed: ParsedWorkflow) -> WorkflowClassification:
        """
        Classify workflow based on parsed AST information.
        """

        quantum_imports = self._detect_quantum_imports(parsed.imports)
        quantum_calls = self._detect_quantum_calls(parsed.function_calls)
        classical_imports = self._detect_classical_imports(parsed.imports)

        # Decision logic
        if not quantum_imports and not quantum_calls:
            workflow_type = WorkflowType.CLASSICAL

        elif quantum_imports or quantum_calls:
            if parsed.has_loops:
                workflow_type = WorkflowType.HYBRID
            else:
                workflow_type = WorkflowType.QUANTUM

        else:
            workflow_type = WorkflowType.CLASSICAL

        return WorkflowClassification(
            workflow_type=workflow_type,
            quantum_imports=quantum_imports,
            quantum_calls=quantum_calls,
            classical_imports=classical_imports,
            has_loops=parsed.has_loops,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _detect_quantum_imports(self, imports: List[str]) -> List[str]:
        detected = []
        for imp in imports:
            base = imp.split(".")[0]
            if any(base == prefix for prefix in self.QUANTUM_IMPORT_PREFIXES):
                detected.append(imp)
        return detected

    def _detect_quantum_calls(self, calls: List[str]) -> List[str]:
        """
        Detect quantum calls STRICTLY:
        - Sampler.run
        - Estimator.run
        - Session.run
        - QuantumCircuit(...)
        - execute(...)
        """
        detected = []

        for call in calls:
            parts = call.split(".")

            # Detect Sampler.run, Estimator.run, Session.run
            if len(parts) == 2:
                obj, method = parts
                if obj in ["Sampler", "Estimator", "Session"] and method == "run":
                    detected.append(call)
                    continue

            # Detect direct quantum API calls
            base = parts[0]
            if any(base == prefix for prefix in self.QUANTUM_CALL_PREFIXES):
                detected.append(call)

        return detected

    def _detect_classical_imports(self, imports: List[str]) -> List[str]:
        detected = []
        for imp in imports:
            base = imp.split(".")[0]
            if any(base == prefix for prefix in self.CLASSICAL_IMPORT_PREFIXES):
                detected.append(imp)
        return detected
