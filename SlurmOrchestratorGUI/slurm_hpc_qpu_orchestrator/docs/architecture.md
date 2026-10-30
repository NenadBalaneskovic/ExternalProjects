# ⭐ `architecture.md` (drop‑in file)

```markdown
# Slurm HPC–QPU Workflow Orchestrator — System Architecture

This document describes the internal architecture of the **Slurm HPC–QPU Workflow Orchestrator**, including its core modules, GUI design, template engine, safety model, and extensibility strategy.  
The orchestrator is built for **static analysis** of Python workflows and **deterministic Slurm script generation** for hybrid HPC–QPU workloads.

---

## 1. Architectural Overview

The system is composed of three major layers:

1. **Core Layer**  
   Static analysis, workflow classification, Slurm template generation.

2. **GUI Layer**  
   PySimpleGUI interface for workflow upload, analysis preview, and script generation.

3. **Template + Theme Layer**  
   Slurm templates, theme JSON, and GUI styling.

The architecture is intentionally **modular**, **testable**, and **safe** — no workflow execution occurs at any point.

---

## 2. Core Layer Architecture

The core layer is responsible for all logic:

```
core/
    ast_parser.py
    workflow_classifier.py
    slurm_template_engine.py
    validators.py
    templates/
    themes/
```

### 2.1 AST Parsing

The **ASTParser** performs static analysis:

- imports  
- function calls  
- loop constructs  
- quantum indicators  
- file metadata  

It produces a structured `ParsedWorkflow` object.

### 2.2 Workflow Classification

The **WorkflowClassifier** determines workflow type:

- `CLASSICAL`  
- `QUANTUM`  
- `HYBRID`

Classification is based on:

- import signatures  
- call signatures  
- quantum library usage  
- hybrid patterns  

### 2.3 Slurm Template Engine

The **SlurmTemplateEngine** loads `.slurm` templates and performs deterministic placeholder substitution.

Templates live in:

```
core/templates/
    classical.slurm
    quantum.slurm
    hybrid.slurm
```

Substitution keys include:

- `JOB_NAME`  
- `PARTITION`  
- `NODES`  
- `CPUS`  
- `TIME_LIMIT`  
- `OUTPUT_LOG`  
- `API_KEY`  
- `RUNTIME_URL`  
- `MODULE_LOAD`  
- `PYTHON_ENV`  

### 2.4 Core Validators

The **core.validators** module ensures:

- valid workflow paths  
- valid substitution dictionaries  
- valid QPU credentials  
- valid workflow types  

---

## 3. GUI Layer Architecture

The GUI layer is fully modular:

```
gui/
    main_gui.py
    components.py
    file_dialogs.py
    theme_manager.py
    styles.css
    utils/
        validators.py
    layout/
        main_layout.py
        workflow_analysis_panel.py
        slurm_preview_panel.py
        credentials_section.py
```

### 3.1 Main Window

The **build_main_window** function constructs:

- workflow upload section  
- analysis panel  
- Slurm preview panel  
- credentials section  
- generate button  

### 3.2 Layout Modules

Each layout component is isolated:

- **workflow_analysis_panel**  
- **slurm_preview_panel**  
- **credentials_section**  
- **upload_section**  

### 3.3 GUI Validators

The GUI layer has its own validator module:

- **gui.utils.validators**

It validates:

- workflow path  
- credential fields  
- GUI input consistency  

### 3.4 Theme Manager

The **ThemeManager** loads JSON themes and applies PySimpleGUI overrides.

CSS variables are mapped for future Qt/WebView migration.

---

## 4. Data Flow

The orchestrator follows a strict, safe pipeline:

```
Workflow File → ASTParser → ParsedWorkflow
ParsedWorkflow → WorkflowClassifier → WorkflowType
WorkflowType + Substitutions → SlurmTemplateEngine → Slurm Script
Slurm Script → GUI Preview → Save to Disk
```

### 4.1 Safety Guarantees

- **No execution of user code**  
- **Static AST analysis only**  
- **Deterministic template substitution**  
- **No dynamic imports**  
- **No eval / exec**  
- **No side effects outside Slurm script generation**

---

## 5. Template Engine Architecture

Templates are stored in:

```
core/templates/
```

Each template is a pure `.slurm` file with placeholders:

```
#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --partition={{PARTITION}}
...
```

The engine performs:

1. load template  
2. validate substitution keys  
3. replace placeholders  
4. return `GeneratedSlurmScript` object  

---

## 6. Extensibility

The orchestrator is designed for future expansion:

### 6.1 Workflow Graph Extraction  
Add a module:

```
core/workflow_graph.py
```

### 6.2 GUI Theme Switching  
Add:

```
gui/layout/theme_switcher.py
```

### 6.3 Export Settings Dialog  
Add:

```
gui/layout/export_settings_dialog.py
```

### 6.4 Additional Slurm Templates  
Add:

```
core/templates/gpu.slurm
core/templates/mpi.slurm
```

---

## 7. Testing Architecture

Tests live in:

```
tests/
```

Coverage includes:

- AST parsing  
- workflow classification  
- template engine  
- GUI layout  
- validators  

Run tests:

```
pytest -q
```

---

## 8. Summary

The Slurm HPC–QPU Workflow Orchestrator is:

- modular  
- deterministic  
- safe  
- testable  
- extensible  

Its architecture cleanly separates:

- core logic  
- GUI layout  
- template engine  
- theming  
- validation  

This ensures long‑term maintainability and scientific reproducibility.

```

---
