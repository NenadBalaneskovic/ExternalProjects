"""
controllers.py

Controller layer for the MemAlloc Optimizer GUI.
Coordinates backend modules and exposes high-level operations
for the GUI (app.py).
"""

from pathlib import Path
from typing import Dict, Optional

from memalloc_core import (
    ScriptManager,
    StaticAnalyzer,
    RuntimeProfiler,
    OptimizationEngine,
    CodeGenerator,
    ExecutionSandbox,
    MetricsStore,
)
from memalloc_core.plots import PlotGenerator


class MemAllocController:
    """
    High-level controller that orchestrates:
    - Script loading
    - Static analysis
    - Optimization planning
    - Code generation
    - Execution sandbox
    - Metrics storage
    - Plot generation
    - Per-run artifact exports (CSV, JSON, logs, flamegraphs)
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        db_path: Path,
        plots_dir: Path,
    ):
        # Backend modules
        self.data_dir = data_dir
        self.script_manager = ScriptManager(data_dir)
        self.analyzer = StaticAnalyzer()
        self.profiler = RuntimeProfiler()
        self.optimizer = OptimizationEngine()
        self.codegen = CodeGenerator(output_dir)
        self.sandbox = ExecutionSandbox()
        self.metrics_store = MetricsStore(db_path)
        self.plots = PlotGenerator(plots_dir)

        # Artifact directory
        self.metrics_dir = data_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)

        # State
        self.loaded_script_path: Optional[Path] = None
        self.loaded_ast = None
        self.analysis_result = None
        self.optimization_plan = None
        self.codegen_result = None
        self.current_script_hash: Optional[str] = None

        # Baseline + optimized results
        self.baseline_runtime = None
        self.optimized_runtime = None

    # --------------------------------------------------------
    # Script loading
    # --------------------------------------------------------

    def load_script(self, script_path: Path) -> Dict:
        result = self.script_manager.load_script(script_path)

        self.loaded_script_path = result.metadata.path
        self.loaded_ast = result.ast_tree
        self.current_script_hash = result.metadata.hash

        return {
            "path": str(script_path),
            "hash": result.metadata.hash,
            "imports": result.metadata.imports,
            "entry_points": result.metadata.entry_points,
            "cached": result.cached,
        }

    # --------------------------------------------------------
    # Static analysis
    # --------------------------------------------------------

    def run_analysis(self) -> Dict:
        if not self.loaded_ast:
            return {"error": "No script loaded."}

        self.analysis_result = self.analyzer.analyze(self.loaded_ast)

        return {
            "hotspots": [
                {
                    "line": h.lineno,
                    "type": h.type,
                    "description": h.description,
                }
                for h in self.analysis_result.hotspots
            ],
            "tips": self.analysis_result.memory_tips,
        }

    # --------------------------------------------------------
    # Optimization plan
    # --------------------------------------------------------

    def build_plan(self, user_selection: Dict[str, bool]) -> Dict:
        if not self.analysis_result:
            return {"error": "Run analysis first."}

        self.optimization_plan = self.optimizer.build_plan(
            self.analysis_result,
            user_selection,
        )

        return {
            "strategies": [
                {
                    "name": s.name,
                    "enabled": s.enabled,
                    "description": s.description,
                }
                for s in self.optimization_plan.strategies
            ],
            "notes": self.optimization_plan.notes,
        }

    # --------------------------------------------------------
    # Code generation
    # --------------------------------------------------------

    def generate_code(self) -> Dict:
        if not self.optimization_plan:
            return {"error": "Build optimization plan first."}

        self.codegen_result = self.codegen.generate(
            self.optimization_plan,
            self.loaded_ast,
        )

        return {
            "notes": self.codegen_result.notes,
            "python_generated": self.codegen_result.optimized_python is not None,
            "cython_generated": self.codegen_result.optimized_cython is not None,
        }

    # --------------------------------------------------------
    # Execution + Metrics Storage + Artifact Export
    # --------------------------------------------------------

    def run_baseline(self) -> Dict:
        if not self.loaded_script_path:
            return {"error": "No script loaded."}

        result = self.sandbox.run_script(self.loaded_script_path)

        self.baseline_runtime = result.runtime_seconds

        # Create record
        record = self.metrics_store.create_record(
            script_hash=self.current_script_hash,
            runtime_seconds=result.runtime_seconds,
            peak_memory_mb=result.peak_memory_mb,
            optimized=False,
            speedup=1.0,
            strategy_summary="baseline",
        )

        # Store baseline metrics
        self.metrics_store.insert_metric(
            script_hash=record.script_hash,
            runtime_seconds=record.runtime_seconds,
            peak_memory_mb=record.peak_memory_mb,
            optimized=record.optimized,
            speedup=record.speedup,
            strategy_summary=record.strategy_summary,
        )

        # Export artifacts
        self.metrics_store.export_json(record, self.metrics_dir)
        self.metrics_store.export_csv(record, self.metrics_dir)
        self.metrics_store.export_log(result.stdout, result.stderr, self.metrics_dir, record)

        return {
            "success": result.success,
            "runtime": result.runtime_seconds,
            "memory": result.peak_memory_mb,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def run_optimized(self) -> Dict:
        assert self.current_script_hash is not None
        optimized_path = self.codegen.output_dir / "optimized_module.py"

        if not optimized_path.exists():
            return {"error": "Optimized module not generated."}

        result = self.sandbox.run_script(optimized_path)

        self.optimized_runtime = result.runtime_seconds

        # Compute speedup
        speedup = (
            self.baseline_runtime / self.optimized_runtime
            if self.baseline_runtime and self.optimized_runtime
            else 1.0
        )

        # Strategy summary
        strategy_summary = ", ".join(
            s.name for s in self.optimization_plan.strategies if s.enabled
        )

        # Create record
        record = self.metrics_store.create_record(
            script_hash=self.current_script_hash,
            runtime_seconds=result.runtime_seconds,
            peak_memory_mb=result.peak_memory_mb,
            optimized=True,
            speedup=speedup,
            strategy_summary=strategy_summary,
        )

        # Store optimized metrics
        self.metrics_store.insert_metric(
            script_hash=record.script_hash,
            runtime_seconds=record.runtime_seconds,
            peak_memory_mb=record.peak_memory_mb,
            optimized=record.optimized,
            speedup=record.speedup,
            strategy_summary=record.strategy_summary,
        )

        # Export artifacts
        self.metrics_store.export_json(record, self.metrics_dir)
        self.metrics_store.export_csv(record, self.metrics_dir)
        self.metrics_store.export_log(result.stdout, result.stderr, self.metrics_dir, record)

        # Optional flamegraph (disabled by default)
        # self.metrics_store.export_flamegraph(lambda: self.sandbox.run_script(optimized_path),
        #                                      self.metrics_dir, record)

        return {
            "success": result.success,
            "runtime": result.runtime_seconds,
            "memory": result.peak_memory_mb,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    # --------------------------------------------------------
    # Metrics retrieval
    # --------------------------------------------------------

    def get_metrics(self) -> Dict:
        return {"metrics": self.metrics_store.get_all_metrics()}

    # --------------------------------------------------------
    # Plot generation
    # --------------------------------------------------------

    def generate_plots(self) -> Dict:
        metrics = self.metrics_store.get_metrics_by_hash(self.current_script_hash)

        if not metrics:
            return {
                "memory_plot": None,
                "runtime_plot": None,
                "speedup_plot": None,
                "error": "No metrics available. Run baseline and optimized execution first."
            }

        paths = self.plots.generate_plots(metrics)

        return {
            "memory_plot": str(paths.memory_plot) if paths.memory_plot else None,
            "runtime_plot": str(paths.runtime_plot) if paths.runtime_plot else None,
            "speedup_plot": str(paths.speedup_plot) if paths.speedup_plot else None,
        }
