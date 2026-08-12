#!/usr/bin/env python3
"""
export_templates.py
-------------------
Command-line tool for exporting the Slurm template library into a user-specified
directory. Useful for packaging, debugging, or distributing template files.

This script:
    - loads template paths from core/template_library
    - copies them into an output directory
    - prints a summary of exported templates

It NEVER executes user code.
"""

import argparse
import shutil
from pathlib import Path

from core import (
    CLASSICAL_TEMPLATE_PATH,
    QUANTUM_TEMPLATE_PATH,
    HYBRID_TEMPLATE_PATH,
)


def print_header(title: str):
    print("=" * 70)
    print(title)
    print("=" * 70)


def export_template(src: Path, dst_dir: Path):
    """
    Copy a template file to the destination directory.
    """
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def main():
    parser = argparse.ArgumentParser(
        description="Export Slurm template library to a directory."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./exported_templates",
        help="Directory where templates will be exported."
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("Slurm HPC–QPU Workflow Orchestrator: Template Export")

    # ----------------------------------------------------------------------
    # Export templates
    # ----------------------------------------------------------------------
    exported_files = []

    exported_files.append(export_template(CLASSICAL_TEMPLATE_PATH, output_dir))
    exported_files.append(export_template(QUANTUM_TEMPLATE_PATH, output_dir))
    exported_files.append(export_template(HYBRID_TEMPLATE_PATH, output_dir))

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print("Templates exported:")
    for f in exported_files:
        print(f"  - {f}")

    print()
    print_header("Export Complete")


if __name__ == "__main__":
    main()