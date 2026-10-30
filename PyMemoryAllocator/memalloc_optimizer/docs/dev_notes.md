# **Developer Notes — MemAlloc Optimizer**

This document contains internal notes for developers working on the MemAlloc Optimizer.  
It supplements `architecture.md` and `usage.md` with practical guidance, conventions, and implementation details.

---

## **1. Core Development Principles**

The project follows these principles:

- **Modularity** — GUI and Core are strictly separated.
- **Reproducibility** — deterministic analysis, profiling, and metrics.
- **Transparency** — all decisions (hotspots, strategies, notes) must be inspectable.
- **Safety** — execution sandbox isolates user code.
- **Extensibility** — new strategies, analyzers, or GUI panels should be easy to add.

---

## **2. Repository Layout**

```
memalloc_optimizer/
│
├── memalloc_core/        # Backend logic
├── memalloc_gui/         # GUI application
├── examples/             # Synthetic workloads
├── tests/                # Full pytest suite
│
├── requirements.txt
├── pyproject.toml
├── setup.cfg
└── README.md
```

---

## **3. Coding Conventions**

### **3.1 Python Version**
- Python **3.12+** required.
- Avoid deprecated modules (e.g., `distutils`).

### **3.2 Style**
- Follow PEP‑8.
- Use type hints everywhere.
- Avoid implicit imports.
- Prefer pure functions in core modules.

### **3.3 Error Handling**
- Core modules must **never raise raw exceptions** to the GUI.
- Always wrap errors in structured result objects:
  - `ProfileResult`
  - `AnalysisResult`
  - `OptimizationPlan`
  - `ExecutionResult`
  - `CodegenResult`

---

## **4. Static Analysis Notes**

### **4.1 AST Parsing**
- Use `ast.parse()` only once per script.
- Cache results in `analysis_cache.json`.

### **4.2 Hotspot Detection**
Hotspot types:
- `temporary_array`
- `repeated_allocation`
- `nested_loop`
- `large_allocation`

Detection rules must be:
- deterministic  
- conservative  
- explainable  

---

## **5. Runtime Profiling Notes**

### **5.1 Function Profiling**
Uses:
- `tracemalloc`
- peak memory
- allocation count
- top snapshot lines

### **5.2 Script Profiling**
Uses:
- subprocess execution
- psutil RSS tracking

Important:
- profiling must not hang  
- always enforce timeouts  
- capture stdout/stderr  

---

## **6. Optimization Engine Notes**

### **6.1 Strategy Requirements**
Automatic requirements:
- temporary arrays → Cython memoryviews  
- nested loops → Numba JIT  
- repeated allocations → preallocation  
- large allocations → layout optimization  

### **6.2 Strategy Conflicts**
Avoid enabling:
- Cython + Numba simultaneously for the same function  
- layout optimization when shapes are dynamic  

---

## **7. Code Generation Notes**

### **7.1 Python Codegen**
Always generated.

### **7.2 Cython Codegen**
Generated only if:
- `cython_required = True`
- user enabled `cython_memoryviews`

### **7.3 Output Directory**
All generated files go into:

```
generated/
```

### **7.4 Compilation**
GUI handles Cython compilation automatically.

---

## **8. Execution Sandbox Notes**

### **8.1 Safety**
Sandbox must:
- isolate user code  
- prevent GUI freeze  
- capture stdout/stderr  
- enforce timeouts  

### **8.2 Baseline vs Optimized**
Both must produce:
- runtime  
- peak memory  
- success flag  
- error message  

---

## **9. Metrics Store Notes**

### **9.1 DuckDB Schema**
Columns:
- script_hash  
- runtime_seconds  
- peak_memory_mb  
- speedup  
- strategy_summary  
- timestamp  

### **9.2 Aggregation**
Useful queries:
- best speedup per script  
- average memory usage  
- strategy effectiveness  

---

## **10. Plot Generation Notes**

Plots include:
- memory usage  
- runtime comparison  
- speedup curves  

Stored in:

```
plots/
```

---

## **11. Testing Notes**

### **11.1 Test Suite Structure**
```
tests/
    test_static_analysis.py
    test_runtime_profiler.py
    test_optimization_engine.py
    test_codegen.py
    test_execution_sandbox.py
    test_metrics_store.py
    test_integration_gui_core.py
```

### **11.2 Guidelines**
- No external dependencies  
- No network access  
- Use temporary directories  
- Use synthetic workloads  
- Tests must be deterministic  

---

## **12. Performance Notes**

### **12.1 Profiling**
Use:
- `cProfile`
- `tracemalloc`
- `psutil`

### **12.2 Optimization**
Focus on:
- reducing allocations  
- improving data layout  
- accelerating nested loops  
- minimizing Python overhead  

---

## **13. Future Extensions**

Potential enhancements:
- GPU acceleration (CuPy)
- Numba parallel loops
- memory layout visualizer
- JupyterLab extension
- web dashboard (Plotly Dash)
- arena allocator simulation

---
