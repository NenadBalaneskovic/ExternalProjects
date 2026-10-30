# `workflow_detection.md`

```markdown
# Workflow Detection — Slurm HPC–QPU Workflow Orchestrator

This document describes the workflow‑detection subsystem of the orchestrator.  
Workflow detection is responsible for identifying whether a Python workflow is:

- **CLASSICAL**
- **QUANTUM**
- **HYBRID**

Detection is performed entirely through **static AST analysis** — no workflow execution occurs at any point.

---

## 1. Detection Pipeline Overview

Workflow detection follows a deterministic pipeline:

```
Workflow File → ASTParser → ParsedWorkflow
ParsedWorkflow → WorkflowClassifier → WorkflowType
WorkflowType → Slurm Template Selection
```

The pipeline is composed of two core modules:

- **ASTParser**  
- **WorkflowClassifier**  

Both operate exclusively on static information.

---

## 2. AST Parsing

The **ASTParser** extracts structural information from the workflow:

### 2.1 Extracted Signals

- **Imports**  
  e.g., `import qiskit`, `from qiskit_ibm_runtime import QiskitRuntimeService`

- **Function Calls**  
  e.g., `QuantumCircuit()`, `execute()`, `backend.run()`

- **Loop Constructs**  
  e.g., `for`, `while`, nested loops

- **Quantum Indicators**  
  e.g., `AerSimulator`, `QiskitRuntimeService`, `transpile`

- **Classical HPC Indicators**  
  e.g., heavy loops, numerical libraries

The parser produces a `ParsedWorkflow` object containing:

- `imports`  
- `function_calls`  
- `has_loops`  
- `quantum_imports`  
- `quantum_calls`  
- `classical_imports`  

This object is passed to the classifier.

---

## 3. Workflow Classification

The **WorkflowClassifier** determines workflow type using a rule‑based system.

### 3.1 Classical Detection

A workflow is **CLASSICAL** if:

- no quantum imports are present  
- no quantum calls are present  
- loops or numerical libraries dominate the structure  

Classical indicators include:

- `numpy`, `scipy`, `matplotlib`, `sympy`  
- heavy loop usage  
- absence of Qiskit/Qutip imports  

### 3.2 Quantum Detection

A workflow is **QUANTUM** if:

- quantum imports are present  
- quantum calls are present  
- no classical HPC loops dominate the structure  

Quantum indicators include:

- `qiskit`  
- `qiskit_ibm_runtime`  
- `qiskit_aer`  
- `qutip`  

Quantum calls include:

- `QuantumCircuit()`  
- `execute()`  
- `backend.run()`  
- `QiskitRuntimeService()`  

### 3.3 Hybrid Detection

A workflow is **HYBRID** if:

- quantum imports **and** classical HPC loops coexist  
- quantum calls appear inside or near loops  
- numerical libraries are used alongside Qiskit/Qutip  

Hybrid indicators include:

- loops around quantum execution  
- classical preprocessing + quantum execution  
- iterative quantum sampling  
- hybrid variational algorithms (VQE, QAOA)  

Hybrid detection is conservative:  
a workflow is only marked HYBRID if both classical and quantum signals are strong.

---

## 4. Detection Heuristics

The classifier uses a multi‑signal heuristic system:

### 4.1 Import‑Based Heuristics

Quantum imports → quantum or hybrid  
Classical imports → classical or hybrid

### 4.2 Call‑Based Heuristics

Quantum calls → quantum or hybrid  
Classical numerical calls → classical or hybrid

### 4.3 Loop‑Based Heuristics

Loops + quantum calls → hybrid  
Loops only → classical  
Quantum calls only → quantum

### 4.4 Dominance Heuristics

If quantum signals dominate → quantum  
If classical signals dominate → classical  
If both dominate → hybrid

---

## 5. Safety Model

Workflow detection is designed for maximum safety:

### 5.1 No Execution of User Code

The system never:

- imports workflow modules  
- executes workflow functions  
- evaluates workflow strings  
- loads dynamic code  

### 5.2 Static AST Analysis Only

All detection is based on:

- Python AST  
- static import detection  
- static call detection  
- structural heuristics  

### 5.3 Deterministic Behavior

Detection is:

- reproducible  
- predictable  
- environment‑independent  

---

## 6. Integration with Slurm Template Engine

Workflow type determines which template is used:

| Workflow Type | Template |
|---------------|----------|
| CLASSICAL     | `classical.slurm` |
| QUANTUM       | `quantum.slurm` |
| HYBRID        | `hybrid.slurm` |

Template selection is performed by:

- **SlurmTemplateEngine**

---

## 7. Extensibility

The detection system is designed for future expansion.

### 7.1 Additional Workflow Types

You can add:

- GPU workflows  
- MPI workflows  
- Array jobs  
- Distributed quantum workflows  

### 7.2 Additional Quantum Indicators

Add new quantum libraries:

- `braket`  
- `pennylane`  
- `cirq`  

### 7.3 Additional Classical Indicators

Add HPC libraries:

- `mpi4py`  
- `cupy`  
- `jax`  

### 7.4 Graph‑Based Detection

Future module:

```
core/workflow_graph.py
```

This will allow:

- control‑flow graph extraction  
- data‑flow analysis  
- hybrid region detection  

---

## 8. Testing

Detection tests live in:

```
tests/test_workflow_classifier.py
```

Tests cover:

- import detection  
- call detection  
- loop detection  
- quantum indicators  
- hybrid heuristics  
- template selection  

Run tests:

```
pytest -q
```

---

## 9. Summary

The workflow‑detection subsystem is:

- modular  
- deterministic  
- safe  
- extensible  
- fully integrated with AST analysis and Slurm template generation  

It provides a robust foundation for classifying classical, quantum, and hybrid workloads.

```

---
