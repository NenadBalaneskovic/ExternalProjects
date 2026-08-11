# **MemAlloc Optimizer**
A scientific memory‑optimization toolkit for Python.  
It combines **static analysis**, **runtime profiling**, **optimization strategy planning**, **code generation**, **execution sandboxing**, **metrics storage**, and a full **PySimpleGUI desktop application**.

The goal: help developers understand and optimize memory‑intensive Python workloads — from numerical pipelines to ML preprocessing, simulation loops, and large‑scale data processing.

---

## **Features**
- **Static Analysis**  
  Detects memory hotspots such as:
  - temporary allocations  
  - repeated allocations  
  - nested loops  
  - large array allocations  

- **Runtime Profiling**  
  Measures:
  - peak memory usage  
  - runtime  
  - allocation count  
  - top memory snapshot lines  

- **Optimization Engine**  
  Builds a strategy plan using:
  - Cython memoryviews  
  - Numba JIT  
  - preallocation  
  - layout optimization  

- **Code Generation**  
  Produces optimized Python and optional Cython modules.

- **Execution Sandbox**  
  Safely runs baseline and optimized scripts.

- **Metrics Store (DuckDB)**  
  Stores performance metrics for long‑term analysis.

- **Plot Generator**  
  Creates memory/runtime/speedup plots.

- **Full GUI Application**  
  Built with PySimpleGUI, modular, themeable, and reproducible.

---

## **Project Structure**
```
memalloc_optimizer/
│
├── memalloc_core/        # Backend: analysis, profiling, codegen, execution
├── memalloc_gui/         # Frontend: GUI, controllers, theming, layout
├── examples/             # Synthetic and real-world workloads
├── tests/                # Full pytest suite
│
├── pyproject.toml
├── setup.cfg
├── requirements.txt
└── README.md
```

---

## **Installation**

### **1. Clone the repository**
```
git clone https://github.com/yourname/memalloc-optimizer.git
cd memalloc-optimizer
```

### **2. Install dependencies**
```
pip install -r requirements.txt
```

### **3. Install the package (editable mode)**
```
pip install -e .
```

---

## **Running the GUI**
Launch the desktop application:

```
memalloc
```

or:

```
python -m memalloc_gui.app
```

---

## **Using the Core API (without GUI)**

### **Static Analysis**
```python
from memalloc_core.static_analysis import StaticAnalyzer
import ast

tree = ast.parse(open("script.py").read())
result = StaticAnalyzer().analyze(tree)

print(result.hotspots)
print(result.memory_tips)
```

### **Runtime Profiling**
```python
from memalloc_core.runtime_profiler import RuntimeProfiler

profiler = RuntimeProfiler()
result = profiler.profile_script("script.py")

print(result.runtime_seconds, result.peak_memory_mb)
```

### **Optimization Plan**
```python
from memalloc_core.optimization_engine import OptimizationEngine

plan = OptimizationEngine().build_plan(analysis_result, user_selection={})
print(plan.strategies)
```

### **Code Generation**
```python
from memalloc_core.codegen import CodeGenerator

cg = CodeGenerator("generated/")
cg.generate(plan, tree)
```

---

## **Examples**
The `examples/` directory contains ready‑to‑run workloads:

- **ranking_script.py** — pairwise ranking with nested loops  
- **ml_pipeline.py** — synthetic ML preprocessing + logistic regression  
- **synthetic_benchmark.py** — full pipeline benchmark across multiple workloads  

Run any example:

```
python examples/ml_pipeline.py
```

---

## **Testing**
Run the full test suite:

```
pytest -q
```

Includes:

- static analysis tests  
- runtime profiler tests  
- optimization engine tests  
- codegen tests  
- execution sandbox tests  
- integration tests (GUI ↔ core)  

---

## **Development Workflow**
- Clean modular architecture (MVC‑style)
- Reproducible scientific workflows
- DuckDB for local metrics storage
- Cython + Numba optional acceleration
- PySimpleGUI for stable cross‑platform GUI

---

## **License**
MIT License — see `LICENSE` for details.

---
