#!/usr/bin/env python3
"""
analyze_workflow.py
-------------------
Command-line tool for analyzing Python workflow files using the
Slurm HPC–QPU Workflow Orchestrator core modules.

This script:
    - parses the workflow using ASTParser
    - classifies it using WorkflowClassifier
    - prints a structured analysis report

It NEVER executes user code.
"""

import argparse
from pathlib import Path

from core import ASTParser, WorkflowClassifier, WorkflowType


def print_header(title: str):
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a Python workflow file (static AST inspection)."
    )
    parser.add_argument(
        "workflow_file",
        type=str,
        help="Path to the Python workflow file to analyze."
    )

    args = parser.parse_args()
    file_path = Path(args.workflow_file)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    print_header("Slurm HPC–QPU Workflow Orchestrator: Static Workflow Analysis")

    # ----------------------------------------------------------------------
    # Parse workflow
    # ----------------------------------------------------------------------
    ast_parser = ASTParser()
    parsed = ast_parser.parse_file(file_path)

    print("Parsed Workflow:")
    print(f"  File: {parsed.file_path}")
    print(f"  Imports: {parsed.imports}")
    print(f"  Function Calls: {parsed.function_calls}")
    print(f"  Contains Loops: {parsed.has_loops}")
    print()

    # ----------------------------------------------------------------------
    # Classify workflow
    # ----------------------------------------------------------------------
    classifier = WorkflowClassifier()
    classification = classifier.classify(parsed)

    print_header("Workflow Classification Result")

    workflow_type = classification.workflow_type

    if workflow_type == WorkflowType.CLASSICAL:
        print("Workflow Type: CLASSICAL (HPC-only)")
    elif workflow_type == WorkflowType.QUANTUM:
        print("Workflow Type: QUANTUM (QPU-only)")
    elif workflow_type == WorkflowType.HYBRID:
        print("Workflow Type: HYBRID (HPC + QPU)")
    else:
        print("Workflow Type: UNKNOWN")

    print()
    print("Detected Quantum Imports:", classification.quantum_imports)
    print("Detected Quantum Calls:", classification.quantum_calls)
    print("Detected Classical Imports:", classification.classical_imports)
    print("Loop Detected:", classification.has_loops)
    print()

    print_header("Analysis Complete")


if __name__ == "__main__":
    main()