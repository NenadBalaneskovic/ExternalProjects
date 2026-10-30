"""
Unit tests for SlurmTemplateEngine.

These tests verify:
    - correct template selection based on WorkflowType
    - correct placeholder substitution
    - correct output file creation
    - correct SlurmScript metadata
    - safe static behavior (no execution of user code)

All tests use temporary directories and never execute user code.
"""

import tempfile
from pathlib import Path

from core.slurm_template_engine import SlurmTemplateEngine
from core.workflow_classifier import WorkflowType
from core.template_library import (
    CLASSICAL_TEMPLATE_PATH,
    QUANTUM_TEMPLATE_PATH,
    HYBRID_TEMPLATE_PATH,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def make_temp_dir() -> Path:
    """Create a temporary directory and return its Path."""
    tmp = tempfile.TemporaryDirectory()
    return Path(tmp.name)


def read(path: Path) -> str:
    """Read a file as UTF-8 text."""
    return path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_classical_template_selection():
    out_dir = make_temp_dir()
    engine = SlurmTemplateEngine(out_dir)

    subs = {
        "JOB_NAME": "test_job",
        "PARTITION": "compute",
        "NODES": "1",
        "CPUS": "4",
        "TIME_LIMIT": "01:00:00",
        "OUTPUT_LOG": "logs/out.log",
        "API_KEY": "{{API_KEY}}",
        "RUNTIME_URL": "{{RUNTIME_URL}}",
        "MODULE_LOAD": "python/3.10",
        "PYTHON_ENV": "/venv",
    }

    script = engine.generate_slurm_script(
        workflow_type=WorkflowType.CLASSICAL,
        substitutions=subs,
        script_name="workflow.py"
    )

    assert script.template_used == CLASSICAL_TEMPLATE_PATH
    assert script.output_path.exists()
    assert "python workflow.py" in script.script_text


def test_quantum_template_selection():
    out_dir = make_temp_dir()
    engine = SlurmTemplateEngine(out_dir)

    subs = {
        "JOB_NAME": "quantum_job",
        "PARTITION": "qpu",
        "NODES": "1",
        "CPUS": "1",
        "TIME_LIMIT": "00:30:00",
        "OUTPUT_LOG": "logs/qpu.log",
        "API_KEY": "{{API_KEY}}",
        "RUNTIME_URL": "{{RUNTIME_URL}}",
        "MODULE_LOAD": "python/3.10",
        "PYTHON_ENV": "/venv",
    }

    script = engine.generate_slurm_script(
        workflow_type=WorkflowType.QUANTUM,
        substitutions=subs,
        script_name="quantum_workflow.py"
    )

    assert script.template_used == QUANTUM_TEMPLATE_PATH
    assert script.output_path.exists()
    assert "QISKIT_RUNTIME_API_KEY" in script.script_text
    assert "python quantum_workflow.py" in script.script_text


def test_hybrid_template_selection():
    out_dir = make_temp_dir()
    engine = SlurmTemplateEngine(out_dir)

    subs = {
        "JOB_NAME": "hybrid_job",
        "PARTITION": "compute",
        "NODES": "1",
        "CPUS": "8",
        "TIME_LIMIT": "02:00:00",
        "OUTPUT_LOG": "logs/hybrid.log",
        "API_KEY": "{{API_KEY}}",
        "RUNTIME_URL": "{{RUNTIME_URL}}",
        "MODULE_LOAD": "python/3.10",
        "PYTHON_ENV": "/venv",
    }

    script = engine.generate_slurm_script(
        workflow_type=WorkflowType.HYBRID,
        substitutions=subs,
        script_name="hybrid_workflow.py"
    )

    assert script.template_used == HYBRID_TEMPLATE_PATH
    assert script.output_path.exists()
    assert "QISKIT_RUNTIME_API_KEY" in script.script_text
    assert "python hybrid_workflow.py" in script.script_text


def test_placeholder_substitution():
    out_dir = make_temp_dir()
    engine = SlurmTemplateEngine(out_dir)

    subs = {
        "JOB_NAME": "placeholder_test",
        "PARTITION": "compute",
        "NODES": "2",
        "CPUS": "16",
        "TIME_LIMIT": "03:00:00",
        "OUTPUT_LOG": "logs/test.log",
        "API_KEY": "MY_KEY",
        "RUNTIME_URL": "https://runtime",
        "MODULE_LOAD": "python/3.10",
        "PYTHON_ENV": "/env",
    }

    script = engine.generate_slurm_script(
        workflow_type=WorkflowType.CLASSICAL,
        substitutions=subs,
        script_name="test.py"
    )

    text = script.script_text

    assert "placeholder_test" in text
    assert "compute" in text
    assert "2" in text
    assert "16" in text
    assert "03:00:00" in text
    assert "logs/test.log" in text
    assert "MY_KEY" in text
    assert "https://runtime" in text
    assert "python test.py" in text


def test_output_file_written_correctly():
    out_dir = make_temp_dir()
    engine = SlurmTemplateEngine(out_dir)

    subs = {
        "JOB_NAME": "write_test",
        "PARTITION": "compute",
        "NODES": "1",
        "CPUS": "4",
        "TIME_LIMIT": "01:00:00",
        "OUTPUT_LOG": "logs/out.log",
        "API_KEY": "{{API_KEY}}",
        "RUNTIME_URL": "{{RUNTIME_URL}}",
        "MODULE_LOAD": "python/3.10",
        "PYTHON_ENV": "/venv",
    }

    script = engine.generate_slurm_script(
        workflow_type=WorkflowType.CLASSICAL,
        substitutions=subs,
        script_name="write_test.py"
    )

    assert script.output_path.exists()
    assert read(script.output_path) == script.script_text