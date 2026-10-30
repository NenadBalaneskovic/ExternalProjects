"""
template_library.py
-------------------
Defines absolute paths to Slurm templates used by the orchestrator.

These paths are imported by SlurmTemplateEngine and GUI preview modules.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# Your actual folder is template_library/, not templates/
CLASSICAL_TEMPLATE_PATH = BASE_DIR / "template_library" / "classical_template.slurm"
QUANTUM_TEMPLATE_PATH   = BASE_DIR / "template_library" / "quantum_template.slurm"
HYBRID_TEMPLATE_PATH    = BASE_DIR / "template_library" / "hybrid_template.slurm"

def validate_template_paths():
    missing = []
    for p in [
        CLASSICAL_TEMPLATE_PATH,
        QUANTUM_TEMPLATE_PATH,
        HYBRID_TEMPLATE_PATH,
    ]:
        if not p.exists():
            missing.append(str(p))
    return missing


