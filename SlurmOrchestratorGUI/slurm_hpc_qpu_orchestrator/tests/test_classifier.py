"""
Unit tests for WorkflowClassifier.

These tests verify:
    - quantum import detection
    - quantum call detection
    - classical import detection
    - hybrid loop detection
    - correct workflow type classification
    - correct WorkflowClassification structure

All tests use temporary files and never execute user code.
"""

import tempfile
from pathlib import Path

from core.ast_parser import ASTParser
from core.workflow_classifier import (
    WorkflowClassifier,
    WorkflowType,
    WorkflowClassification,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def write_temp_python(code: str) -> Path:
    """Write code to a temporary Python file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
    tmp.write(code.encode("utf-8"))
    tmp.close()
    return Path(tmp.name)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_classical_workflow_detection():
    code = """
import numpy as np
import scipy
def f():
    return np.sum([1, 2, 3])
"""
    path = write_temp_python(code)
    parser = ASTParser()
    parsed = parser.parse_file(path)

    classifier = WorkflowClassifier()
    result = classifier.classify(parsed)

    assert result.workflow_type == WorkflowType.CLASSICAL
    assert "numpy" in result.classical_imports
    assert result.quantum_imports == []
    assert result.quantum_calls == []
    assert result.has_loops is False


def test_quantum_workflow_detection():
    code = """
import qiskit_ibm_runtime
def f():
    sampler = qiskit_ibm_runtime.Sampler()
    sampler.run()
"""
    path = write_temp_python(code)
    parser = ASTParser()
    parsed = parser.parse_file(path)

    classifier = WorkflowClassifier()
    result = classifier.classify(parsed)

    assert result.workflow_type == WorkflowType.QUANTUM
    assert "qiskit_ibm_runtime" in result.quantum_imports
    assert "qiskit_ibm_runtime.Sampler" in result.quantum_calls or "Sampler" in result.quantum_calls
    assert result.has_loops is False


def test_hybrid_workflow_detection():
    code = """
import qiskit_ibm_runtime
import numpy as np

def hybrid():
    sampler = qiskit_ibm_runtime.Sampler()
    for i in range(5):
        sampler.run()
"""
    path = write_temp_python(code)
    parser = ASTParser()
    parsed = parser.parse_file(path)

    classifier = WorkflowClassifier()
    result = classifier.classify(parsed)

    assert result.workflow_type == WorkflowType.HYBRID
    assert "qiskit_ibm_runtime" in result.quantum_imports
    assert result.has_loops is True


def test_no_quantum_no_classical_defaults_to_classical():
    code = """
def f():
    print("hello")
"""
    path = write_temp_python(code)
    parser = ASTParser()
    parsed = parser.parse_file(path)

    classifier = WorkflowClassifier()
    result = classifier.classify(parsed)

    assert result.workflow_type == WorkflowType.CLASSICAL
    assert result.quantum_imports == []
    assert result.quantum_calls == []
    assert result.classical_imports == []
    assert result.has_loops is False


def test_workflowclassification_structure():
    code = "import numpy"
    path = write_temp_python(code)
    parser = ASTParser()
    parsed = parser.parse_file(path)

    classifier = WorkflowClassifier()
    result = classifier.classify(parsed)

    assert isinstance(result, WorkflowClassification)
    assert isinstance(result.quantum_imports, list)
    assert isinstance(result.quantum_calls, list)
    assert isinstance(result.classical_imports, list)
    assert isinstance(result.has_loops, bool)
    assert isinstance(result.workflow_type, WorkflowType)