#!/usr/bin/env python3
"""
generate_slurm.py
-----------------
Command-line tool for generating Slurm job scripts using the
Slurm HPC–QPU Workflow Orchestrator core modules.

This script:
    - parses the workflow using ASTParser
    - classifies it using WorkflowClassifier
    - selects the correct Slurm template
    - applies placeholder substitutions
    - writes the final .slurm file

It NEVER executes user code.
"""

import argparse
from pathlib import Path

from core import (
    ASTParser,
    WorkflowClassifier,
    WorkflowType,
    SlurmTemplateEngine,
)


def print_header(title: str):
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Slurm script for a Python workflow file."
    )
    parser.add_argument(
        "workflow_file",
        type=str,
        help="Path to the Python workflow file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./generated_slurm_jobs",
        help="Directory where the Slurm file will be written."
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="my_slurm_job",
        help="Job name for Slurm."
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="compute",
        help="Slurm partition."
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default="1",
        help="Number of nodes."
    )
    parser.add_argument(
        "--cpus",
        type=str,
        default="4",
        help="CPUs per node."
    )
    parser.add_argument(
        "--time",
        type=str,
        default="01:00:00",
        help="Time limit."
    )
    parser.add_argument(
        "--log",
        type=str,
        default="logs/%x_%j.out",
        help="Output log path."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="{{API_KEY}}",
        help="QPU API key placeholder."
    )
    parser.add_argument(
        "--runtime-url",
        type=str,
        default="{{RUNTIME_URL}}",
        help="QPU runtime URL placeholder."
    )
    parser.add_argument(
        "--module-load",
        type=str,
        default="python/3.10",
        help="Module load command."
    )
    parser.add_argument(
        "--venv",
        type=str,
        default="{{PYTHON_ENV}}",
        help="Virtual environment path."
    )

    args = parser.parse_args()
    file_path = Path(args.workflow_file)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    print_header("Slurm HPC–QPU Workflow Orchestrator: Slurm Generation")

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

    print_header("Workflow Classification")
    print(f"Workflow Type: {classification.workflow_type.name}")
    print()

    # ----------------------------------------------------------------------
    # Prepare substitutions
    # ----------------------------------------------------------------------
    substitutions = {
        "JOB_NAME": args.job_name,
        "PARTITION": args.partition,
        "NODES": args.nodes,
        "CPUS": args.cpus,
        "TIME_LIMIT": args.time,
        "OUTPUT_LOG": args.log,
        "API_KEY": args.api_key,
        "RUNTIME_URL": args.runtime_url,
        "MODULE_LOAD": args.module_load,
        "PYTHON_ENV": args.venv,
    }

    # ----------------------------------------------------------------------
    # Generate Slurm script
    # ----------------------------------------------------------------------
    engine = SlurmTemplateEngine(Path(args.output_dir))
    slurm_script = engine.generate_slurm_script(
        workflow_type=classification.workflow_type,
        substitutions=substitutions,
        script_name=file_path.name
    )

    print_header("Slurm Script Generated")
    print(f"Template Used: {slurm_script.template_used}")
    print(f"Output File:   {slurm_script.output_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()