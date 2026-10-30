"""
SlurmTemplateEngine
-------------------
Generates Slurm job scripts based on workflow classification and user-provided
settings. This module NEVER executes user code. It only performs static template
substitution.

Templates are stored in core/template_library/*.slurm and contain placeholders
such as:
    {{JOB_NAME}}, {{PARTITION}}, {{NODES}}, {{CPUS}},
    {{API_KEY}}, {{RUNTIME_URL}}, {{SCRIPT_NAME}}

The engine replaces these placeholders with user-provided values.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from slurm_hpc_qpu_orchestrator.core.template_library import (
    CLASSICAL_TEMPLATE_PATH,
    QUANTUM_TEMPLATE_PATH,
    HYBRID_TEMPLATE_PATH,
)
from .workflow_classifier import WorkflowType


# ---------------------------------------------------------------------------
# Data structure returned by SlurmTemplateEngine
# ---------------------------------------------------------------------------

@dataclass
class SlurmScript:
    script_text: str
    template_used: Path
    output_path: Path


# ---------------------------------------------------------------------------
# Template Engine
# ---------------------------------------------------------------------------

class SlurmTemplateEngine:
    """
    Loads Slurm templates and performs placeholder substitution.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def generate_slurm_script(
        self,
        workflow_type: WorkflowType,
        substitutions: Dict[str, str],
        script_name: str
    ) -> SlurmScript:
        """
        Generate a Slurm script for the given workflow type.

        Parameters
        ----------
        workflow_type : WorkflowType
            CLASSICAL, QUANTUM, or HYBRID.
        substitutions : Dict[str, str]
            Mapping of placeholder → value.
        script_name : str
            Name of the Python workflow file.

        Returns
        -------
        SlurmScript
            Contains the final script text and metadata.
        """

        template_path = self._select_template(workflow_type)
        template_text = template_path.read_text(encoding="utf-8")

        # Always include script name substitution
        substitutions = dict(substitutions)
        substitutions["SCRIPT_NAME"] = script_name

        final_script = self._apply_substitutions(template_text, substitutions)

        out_file = self.output_dir / f"{script_name}.slurm"
        out_file.write_text(final_script, encoding="utf-8")

        return SlurmScript(
            script_text=final_script,
            template_used=template_path,
            output_path=out_file
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _select_template(self, workflow_type: WorkflowType) -> Path:
        """
        Select the correct Slurm template based on workflow type.
        """
        if workflow_type == WorkflowType.CLASSICAL:
            return CLASSICAL_TEMPLATE_PATH
        elif workflow_type == WorkflowType.QUANTUM:
            return QUANTUM_TEMPLATE_PATH
        elif workflow_type == WorkflowType.HYBRID:
            return HYBRID_TEMPLATE_PATH
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

    def _apply_substitutions(self, template: str, subs: Dict[str, str]) -> str:
        """
        Replace {{PLACEHOLDER}} entries in the template with actual values.
        """
        result = template
        for key, value in subs.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, value)
        return result