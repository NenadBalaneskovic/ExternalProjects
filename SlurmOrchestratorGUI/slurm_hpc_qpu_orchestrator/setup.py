"""
setup.py
--------
Packaging configuration for the Slurm HPC–QPU Workflow Orchestrator.

This setup script:
    - installs core modules (AST parser, classifier, template engine)
    - installs GUI modules (PySimpleGUI-based interface)
    - installs CLI entry point: `slurm-orchestrator`
    - includes templates and theme files as package data
"""

from setuptools import setup, find_packages
from pathlib import Path


# ----------------------------------------------------------------------
# Read long description
# ----------------------------------------------------------------------

README = Path(__file__).parent / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else ""


# ----------------------------------------------------------------------
# Package Setup
# ----------------------------------------------------------------------

setup(
    name="slurm-hpc-qpu-orchestrator",
    version="0.1.0",
    description="Workflow analyzer and Slurm script generator for hybrid HPC–QPU workloads.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Nenad",
    url="https://github.com/your-repo-url",
    license="MIT",

    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,

    package_data={
        "core": [
            "templates/*.slurm",
            "themes/*.json",
        ],
        "gui": [
            "styles.css",
        ],
    },

    install_requires=[
        "PySimpleGUI>=4.60",
        "numpy>=1.26",
        "scipy>=1.12",
        "matplotlib>=3.8",
        "sympy>=1.12",
        "qiskit>=1.0.2",
        "qiskit-ibm-runtime>=0.21.0",
    ],

    entry_points={
        "console_scripts": [
            "slurm-orchestrator=core.cli:main",
        ]
    },

    python_requires=">=3.10",
)