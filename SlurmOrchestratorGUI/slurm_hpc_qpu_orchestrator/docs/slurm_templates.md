# `slurm_templates.md`

```markdown
# Slurm Templates — HPC–QPU Workflow Orchestrator

This document describes the Slurm template system used by the orchestrator.  
Templates define how classical, quantum, and hybrid workflows are translated into Slurm job scripts.  
The template engine performs **deterministic placeholder substitution** and guarantees **safe, static behavior**.

---

## 1. Template System Overview

The orchestrator ships with three Slurm templates:

```
core/templates/
    classical.slurm
    quantum.slurm
    hybrid.slurm
```

Each template is a pure `.slurm` file containing placeholders such as:

```
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --partition={{PARTITION}}
#SBATCH --nodes={{NODES}}
#SBATCH --cpus-per-task={{CPUS}}
#SBATCH --time={{TIME_LIMIT}}
```

Templates are loaded and processed by:

- **SlurmTemplateEngine**

The engine never executes workflow code and never performs dynamic imports.

---

## 2. Template Selection Logic

Template selection is based on workflow classification:

| Workflow Type | Template Used |
|---------------|---------------|
| CLASSICAL     | `classical.slurm` |
| QUANTUM       | `quantum.slurm` |
| HYBRID        | `hybrid.slurm` |

Classification is performed by:

- **WorkflowClassifier**

Quantum indicators include imports from:

- `qiskit`
- `qiskit_ibm_runtime`
- `qiskit_aer`
- `qutip`

Hybrid workflows combine classical HPC loops with quantum calls.

---

## 3. Placeholder System

Templates contain placeholders in the form:

```
{{PLACEHOLDER_NAME}}
```

The engine replaces these with values from a substitution dictionary.

### 3.1 Required Placeholders

The following placeholders must be present in every template:

- `JOB_NAME`
- `PARTITION`
- `NODES`
- `CPUS`
- `TIME_LIMIT`
- `OUTPUT_LOG`
- `MODULE_LOAD`
- `PYTHON_ENV`

### 3.2 Quantum‑Specific Placeholders

Quantum templates additionally support:

- `API_KEY`
- `RUNTIME_URL`

These may be real values or placeholders such as:

```
{{API_KEY}}
{{RUNTIME_URL}}
```

GUI validators ensure correctness:

- **gui.utils.validators**

---

## 4. Template Engine Workflow

The template engine follows a deterministic pipeline:

```
Load Template → Validate Keys → Substitute Placeholders → Return Script
```

### 4.1 Loading

Templates are loaded from disk using UTF‑8 encoding.

### 4.2 Validation

The engine checks:

- all required placeholders exist  
- no missing keys  
- no malformed placeholders  

Validation is performed by:

- **core.validators**

### 4.3 Substitution

Substitution is a pure string replacement:

- no eval  
- no exec  
- no dynamic code  
- no workflow execution  

### 4.4 Output

The engine returns a `GeneratedSlurmScript` object containing:

- final script text  
- template name  
- workflow type  
- substitution dictionary  

---

## 5. Template Structure

### 5.1 Classical Template

Classical templates include:

- CPU allocation  
- node count  
- time limit  
- module loading  
- Python environment activation  
- workflow execution command  

### 5.2 Quantum Template

Quantum templates include:

- QPU runtime environment  
- IBM Runtime credentials  
- quantum module loading  
- Python environment activation  

### 5.3 Hybrid Template

Hybrid templates combine:

- classical HPC loops  
- quantum execution blocks  
- environment switching  
- optional QPU credential injection  

---

## 6. Safety Model

The template engine is designed for maximum safety:

### 6.1 No Workflow Execution

The engine never:

- imports workflow modules  
- executes workflow functions  
- evaluates workflow strings  
- loads dynamic code  

### 6.2 Deterministic Substitution

All placeholder replacement is:

- static  
- predictable  
- reproducible  

### 6.3 No Side Effects

The engine only:

- reads template files  
- writes Slurm scripts  
- returns strings  

---

## 7. Extending the Template System

New templates can be added easily:

```
core/templates/gpu.slurm
core/templates/mpi.slurm
core/templates/array.slurm
```

To register a new template:

1. Add the `.slurm` file  
2. Update `WorkflowClassifier` if needed  
3. Add placeholder validation rules  
4. Add GUI support (optional)

### 7.1 Example: GPU Template

Add:

```
#SBATCH --gres=gpu:{{GPU_COUNT}}
```

Then update validators:

- **core.validators**

---

## 8. Testing

Template tests live in:

```
tests/test_template_engine.py
```

Tests cover:

- placeholder validation  
- substitution correctness  
- template selection  
- quantum credential injection  
- hybrid template logic  

Run tests:

```
pytest -q
```

---

## 9. Summary

The Slurm template system is:

- modular  
- deterministic  
- safe  
- extensible  
- fully integrated with AST analysis and workflow classification  

It provides a robust foundation for generating Slurm scripts for classical, quantum, and hybrid workloads.

```

---
