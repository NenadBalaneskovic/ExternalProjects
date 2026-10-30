# **Usage Guide — MemAlloc Optimizer**

This document explains how to **use** the MemAlloc Optimizer:

- Running the GUI  
- Running the CLI  
- Using the Core API  
- Understanding analysis results  
- Generating optimized code  
- Running benchmarks  
- Viewing plots and metrics  

It is intended for developers, researchers, and engineers working with memory‑intensive Python workloads.

---

## **1. Running the GUI**

Launch the desktop application:

```
memalloc
```

or:

```
python -m memalloc_gui.app
```

The GUI provides:

- Script loading  
- Static analysis  
- Optimization plan configuration  
- Code generation  
- Baseline & optimized execution  
- Metrics visualization  
- Plot generation  

The GUI is the recommended interface for most users.

---

## **2. Running the CLI**

We can run the optimizer directly on a script:

```
memalloc path/to/script.py
```

This performs:

1. Static analysis  
2. Optimization plan  
3. Code generation  
4. Baseline execution  
5. Optimized execution  
6. Metrics storage  

CLI output is printed to the terminal and stored in DuckDB.

---

## **3. Using the Core API**

The backend is fully accessible programmatically.

### **3.1 Static Analysis**

```python
from memalloc_core.static_analysis import StaticAnalyzer
import ast

tree = ast.parse(open("script.py").read())
result = StaticAnalyzer().analyze(tree)

print(result.hotspots)
print(result.memory_tips)
```

---

### **3.2 Optimization Plan**

```python
from memalloc_core.optimization_engine import OptimizationEngine

plan = OptimizationEngine().build_plan(result, user_selection={
    "cython_memoryviews": True,
    "numba_jit": True,
    "preallocate_buffers": True,
    "optimize_layout": False,
})

print(plan.strategies)
```

---

### **3.3 Code Generation**

```python
from memalloc_core.codegen import CodeGenerator

cg = CodeGenerator("generated/")
codegen_result = cg.generate(plan, tree)

print(codegen_result.optimized_python)
print(codegen_result.optimized_cython)
```

---

### **3.4 Execution Sandbox**

```python
from memalloc_core.execution_sandbox import ExecutionSandbox

sandbox = ExecutionSandbox()
baseline = sandbox.run_python("script.py")
optimized = sandbox.run_python("generated/optimized.py")

print(baseline.runtime_seconds, optimized.runtime_seconds)
```

---

### **3.5 Metrics Storage**

```python
from memalloc_core.metrics_store import MetricsStore

db = MetricsStore("metrics.duckdb")
db.insert(script_hash="abc123", runtime_seconds=1.23, peak_memory_mb=512, speedup=2.1)
print(db.get_all())
```

---

## **4. Understanding Analysis Results**

Static analysis produces:

- **hotspots**  
- **memory tips**  
- **line numbers**  
- **descriptions**  

Example:

```json
{
  "hotspots": [
    { "lineno": 42, "type": "temporary_array", "description": "Temporary NumPy allocation" },
    { "lineno": 88, "type": "nested_loop", "description": "Quadratic nested loop" }
  ],
  "memory_tips": [
    "Use contiguous arrays",
    "Consider preallocating buffers"
  ]
}
```

---

## **5. Generating Optimized Code**

The optimizer can generate:

- **optimized Python**  
- **optional Cython** (if required)  

Generated files appear in:

```
generated/
    optimized.py
    optimized.pyx
```

We can compile the Cython module manually or let the GUI do it.

---

## **6. Running Benchmarks**

The `examples/` directory contains ready‑to‑run workloads:

- `ranking_script.py`  
- `ml_pipeline.py`  
- `synthetic_benchmark.py`  

Run any example:

```
python examples/ml_pipeline.py
```

---

## **7. Viewing Plots**

Plots are generated automatically:

- memory usage  
- runtime comparison  
- speedup curves  

They are stored in:

```
plots/
    memory_plot.png
    runtime_plot.png
    speedup_plot.png
```

---

## **8. Caching**

The optimizer stores analysis results in:

```
analysis_cache.json
```

This speeds up repeated runs by avoiding re‑analysis.

---

## **9. Development Workflow**

Install in editable mode:

```
pip install -e .
```

Run tests:

```
pytest -q
```

Recommended modules to explore:

- static analysis  
- runtime profiler  
- optimization engine  
- code generation  
- execution sandbox  

---
