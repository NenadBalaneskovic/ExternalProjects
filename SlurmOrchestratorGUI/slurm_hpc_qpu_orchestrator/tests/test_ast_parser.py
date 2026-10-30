"""
Unit tests for ASTParser.

These tests verify:
    - import extraction
    - function call extraction
    - loop detection
    - attribute chain resolution
    - correct ParsedWorkflow structure

All tests use temporary files and never execute user code.
"""

import tempfile
from pathlib import Path

from core.ast_parser import ASTParser, ParsedWorkflow


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

def test_import_extraction():
    code = """
import numpy
import qiskit_ibm_runtime
from scipy import optimize
"""
    path = write_temp_python(code)
    parser = ASTParser()
    parsed = parser.parse_file(path)

    assert "numpy" in parsed.imports
    assert "qiskit_ibm_runtime" in parsed.imports
    assert "scipy" in parsed.imports


def test_function_call_extraction():
    code = """
def f():
    run()
    sampler.run()
    qiskit_ibm_runtime.Sampler()
"""
    path = write_temp_python(code)
    parser = ASTParser()
    parsed = parser.parse_file(path)

    assert "run" in parsed.function_calls
    assert "sampler.run" in parsed.function_calls
    assert "qiskit_ibm_runtime.Sampler" in parsed.function_calls


def test_loop_detection():
    code = """
for i in range(10):
    print(i)

while True:
    break
"""
    path = write_temp_python(code)
    parser = ASTParser()
    parsed = parser.parse_file(path)

    assert parsed.has_loops is True


def test_no_loops():
    code = """
print("no loops here")
"""
    path = write_temp_python(code)
    parser = ASTParser()
    parsed = parser.parse_file(path)

    assert parsed.has_loops is False


def test_parsedworkflow_structure():
    code = "import numpy"
    path = write_temp_python(code)
    parser = ASTParser()
    parsed = parser.parse_file(path)

    assert isinstance(parsed, ParsedWorkflow)
    assert isinstance(parsed.imports, list)
    assert isinstance(parsed.function_calls, list)
    assert isinstance(parsed.has_loops, bool)
    assert parsed.file_path == path