Here is a **complete, production‑ready `architecture.md`**, written specifically for our MemAlloc Optimizer project.  
It is structured, clear, and suitable for GitHub, documentation sites, or internal technical onboarding.

It explains the **full system architecture**, including:

- Core backend  
- GUI frontend  
- Controller layer  
- Data flow  
- Caching  
- Code generation  
- Execution sandbox  
- Metrics store  
- Plotting subsystem  

---

# **MemAlloc Optimizer — System Architecture**

The MemAlloc Optimizer is a modular scientific‑computing system designed to analyze, optimize, and benchmark memory‑intensive Python workloads.  
It consists of two major subsystems:

- **memalloc_core** — backend (analysis, profiling, optimization, codegen, execution, metrics)  
- **memalloc_gui** — frontend (PySimpleGUI application, controllers, theming, layout)

This document describes the architecture, data flow, and responsibilities of each module.

---

## **1. High‑Level Overview**

```
┌──────────────────────────┐
│        GUI Layer         │  PySimpleGUI
│  (memalloc_gui.app)      │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│     Controller Layer     │  MemAllocController
│  (memalloc_gui.controllers) 
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│        Core Layer        │  memalloc_core
│ static_analysis          │
│ runtime_profiler         │
│ optimization_engine      │
│ codegen                  │
│ execution_sandbox        │
│ metrics_store            │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│     Data & Artifacts     │
│ DuckDB metrics           │
│ generated code           │
│ plots                    │
│ analysis cache           │
└──────────────────────────┘
```

---

## **2. Core Architecture (`memalloc_core`)**

### **2.1 Static Analysis**
**Module:** `static_analysis.py`  
**Input:** Python AST  
**Output:** `AnalysisResult`

Responsibilities:
- Parse AST  
- Detect memory hotspots:
  - temporary arrays  
  - repeated allocations  
  - nested loops  
  - large allocations  
- Generate memory optimization tips  
- Provide structured hotspot metadata

### **2.2 Runtime Profiler**
**Module:** `runtime_profiler.py`  
Profiles either:
- Python functions (tracemalloc)  
- Python scripts (subprocess + psutil)

Outputs:
- runtime  
- peak memory  
- allocation count  
- top snapshot lines  
- stdout/stderr  
- success/error state

### **2.3 Optimization Engine**
**Module:** `optimization_engine.py`  
Builds an `OptimizationPlan` based on:
- hotspots  
- user strategy selection  
- automatic strategy requirements

Strategies include:
- Cython memoryviews  
- Numba JIT  
- preallocation  
- layout optimization  

### **2.4 Code Generation**
**Module:** `codegen.py`  
Generates:
- optimized Python module  
- optional Cython module (`.pyx`)  
- notes describing applied transformations

### **2.5 Execution Sandbox**
**Module:** `execution_sandbox.py`  
Safely executes:
- baseline script  
- optimized script  

Captures:
- runtime  
- memory  
- stdout/stderr  
- success/failure

### **2.6 Metrics Store**
**Module:** `metrics_store.py`  
Backend: **DuckDB**  
Stores:
- runtime  
- peak memory  
- speedup  
- strategy summary  
- script hash  

Supports:
- insertion  
- retrieval  
- aggregation  
- export

### **2.7 Plot Generator**
**Module:** `plots.py`  
Creates:
- memory usage plots  
- runtime comparison plots  
- speedup plots  

Outputs PNG files for GUI display.

---

## **3. GUI Architecture (`memalloc_gui`)**

### **3.1 Application Entry Point**
**Module:** `app.py`  
Responsibilities:
- initialize GUI  
- load theme  
- build layout  
- bind events  
- route actions to controller  

### **3.2 Controller Layer**
**Module:** `controllers.py`  
This is the **integration boundary** between GUI and core.

Responsibilities:
- load script  
- compute script hash  
- manage analysis cache  
- run static analysis  
- build optimization plan  
- generate code  
- run baseline & optimized execution  
- store metrics  
- generate plots  
- return structured dictionaries to GUI

### **3.3 Layout System**
**Module:** `layout.py`  
Builds:
- main window  
- tabs  
- panels  
- buttons  
- output areas  

### **3.4 Theming**
**Module:** `theming.py`  
Handles:
- color schemes  
- fonts  
- icons  
- dark/light mode  

### **3.5 View Models**
**Module:** `view_models.py`  
Defines structured data for GUI components:
- ScriptLoadVM  
- AnalysisVM  
- OptimizationPlanVM  
- CodegenVM  
- ExecutionVM  
- MetricsVM  
- PlotsVM  

---

## **4. Data Flow**

### **4.1 Script Loading**
```
GUI → Controller → compute hash → load file → cache lookup
```

### **4.2 Static Analysis**
```
Controller → static_analysis.analyze(AST) → AnalysisResult
```

### **4.3 Optimization Plan**
```
Controller → optimization_engine.build_plan(analysis, user_selection)
```

### **4.4 Code Generation**
```
Controller → codegen.generate(plan, AST) → Python/Cython modules
```

### **4.5 Execution**
```
Controller → execution_sandbox.run_baseline()
Controller → execution_sandbox.run_optimized()
```

### **4.6 Metrics Storage**
```
Controller → metrics_store.insert(...)
```

### **4.7 Plot Generation**
```
Controller → plots.generate(...)
```

---

## **5. Caching**

### **analysis_cache.json**
Stores:
- script hash  
- hotspots  
- memory tips  
- timestamp  
- script path  

Purpose:
- avoid re‑analysis  
- speed up GUI  
- accelerate benchmarks  

---

## **6. Generated Artifacts**

### **generated/**
Contains:
- optimized Python modules  
- optimized Cython modules  
- build logs  
- codegen notes  

### **plots/**
Contains:
- memory plots  
- runtime plots  
- speedup plots  

### **metrics.duckdb**
Contains:
- historical performance data  
- strategy summaries  
- script hashes  

---

## **7. Example Workflows**

### **7.1 GUI Workflow**
1. Load script  
2. Analyze  
3. Build plan  
4. Generate code  
5. Run baseline  
6. Run optimized  
7. Store metrics  
8. View plots  

### **7.2 CLI Workflow**
```bash
memalloc examples/ml_pipeline.py
```

---

## **8. Design Principles**

- **Modularity**  
  Clear separation between GUI and core.

- **Reproducibility**  
  Deterministic analysis, profiling, and metrics.

- **Scientific Transparency**  
  All decisions (hotspots, strategies, notes) are visible.

- **Extensibility**  
  Easy to add new strategies, analyzers, or GUI panels.

- **Safety**  
  Execution sandbox isolates user code.

---

## **9. Future Extensions**

- GPU acceleration (CuPy)  
- Parallel execution (Numba parallel)  
- Memory layout visualizer  
- JupyterLab extension  
- Web dashboard (Plotly Dash)  

---
