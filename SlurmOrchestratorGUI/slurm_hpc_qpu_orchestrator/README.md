# ⭐ `README.md` (drop‑in file)

```markdown
# Slurm HPC–QPU Workflow Orchestrator

A modular workflow analyzer and Slurm script generator for **hybrid HPC–QPU workloads**.  
It statically inspects Python workflows, classifies them (classical, quantum, hybrid), and produces fully‑parameterized Slurm scripts using a safe template engine.  
A PySimpleGUI interface provides an intuitive way to upload workflows, inspect AST structure, preview Slurm scripts, and optionally inject QPU credentials.

---

## ✨ Features

- **Static AST analysis**  
  No workflow execution. Safe inspection of imports, function calls, loops, and quantum indicators.

- **Workflow classification**  
  Automatically detects:
  - classical workflows  
  - QPU workflows (Qiskit, Qiskit Runtime, Qiskit Aer)  
  - hybrid HPC–QPU workflows  

- **Slurm template engine**  
  Generates `.slurm` scripts using:
  - classical templates  
  - quantum templates  
  - hybrid templates  

- **GUI interface**  
  - workflow upload  
  - AST + classification panel  
  - Slurm preview panel  
  - optional QPU credential injection  
  - theme manager  
  - modular layout components  

- **Safe by design**  
  - no execution of user code  
  - static analysis only  
  - deterministic template substitution  

---

## 📦 Project Structure

```
core/
    ast_parser.py
    workflow_classifier.py
    slurm_template_engine.py
    validators.py
    templates/
        classical.slurm
        quantum.slurm
        hybrid.slurm
    themes/
        darkblue3_theme.json

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

tests/
    test_ast_parser.py
    test_workflow_classifier.py
    test_template_engine.py
    test_gui.py

setup.py
requirements.txt
README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```
git clone `https://github.com/your-repo/slurm-hpc-qpu-orchestrator` [(github.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fgithub.com%2Fyour-repo%2Fslurm-hpc-qpu-orchestrator")
cd slurm-hpc-qpu-orchestrator
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Install the package

```
pip install .
```

---

## 🧠 Core Concepts

### Static AST Analysis  
The orchestrator uses a safe AST parser to extract:

- imports  
- function calls  
- loop constructs  
- quantum indicators  
- file metadata  

This is done using:

- **ASTParser**  
- **WorkflowClassifier**  

### Workflow Classification  
Workflows are classified into:

- **CLASSICAL**  
- **QUANTUM**  
- **HYBRID**

Classification is based on imports and call signatures.

### Slurm Template Engine  
The engine loads `.slurm` templates and performs placeholder substitution:

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

Engine entry point:

- **SlurmTemplateEngine**

---

## 🖥 GUI Overview

The GUI is built using PySimpleGUI and provides:

- workflow upload  
- AST + classification panel  
- Slurm preview panel  
- optional QPU credential injection  
- theme manager  
- modular layout components  

Main entry point:

```bash
python -m gui.main_gui
```

Or via CLI:

```bash
slurm-orchestrator workflow.py
```

GUI components:

- **build_main_window**  
- **build_main_layout**  
- **build_workflow_analysis_panel**  
- **build_slurm_preview_panel**  
- **build_credentials_section**  

---

## 🧪 Testing

Run all tests:

```
pytest -q
```

Tests include:

- AST parsing  
- workflow classification  
- template engine  
- GUI layout  
- validators  

---

## 🔧 CLI Usage

Analyze a workflow:

```
slurm-orchestrator my_workflow.py
```

Generate a Slurm script:

```
slurm-orchestrator my_workflow.py --generate
```

---

## 🎨 Theming

Themes are defined in:

```
core/themes/darkblue3_theme.json
```

Applied via:

- **ThemeManager**

CSS variables are mapped for future Qt/WebView migration.

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 🤝 Contributing

Pull requests are welcome.  
Please ensure:

- clean modular design  
- deterministic behavior  
- no execution of user workflows  
- full test coverage  

---
