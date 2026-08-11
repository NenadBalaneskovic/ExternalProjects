"""
app.py — MemAlloc Optimizer GUI
"""

import PySimpleGUI as sg
from pathlib import Path

from memalloc_gui.controllers import MemAllocController
from memalloc_gui.theming import ThemeLoader, apply_theme
from memalloc_gui.view_models import (
    ScriptLoadVM,
    AnalysisVM,
    OptimizationPlanVM,
    CodegenVM,
    ExecutionVM,
    PlotsVM,
)

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "memalloc_data"
PLOTS_DIR = DATA_DIR / "plots"
DB_PATH = DATA_DIR / "metrics.duckdb"
OUTPUT_DIR = DATA_DIR / "generated"

THEME_PATH = BASE_DIR / "assets" / "darkblue3_theme.json"


# ============================================================
# GUI Layout (EVERY ROW IS A LIST)
# ============================================================

def build_main_layout():
    return [

        # Script Loader
        [sg.Text("Upload Python Script:")],
        [
            sg.Input(key="-SCRIPT_PATH-", enable_events=True),
            sg.FileBrowse(file_types=(("Python Files", "*.py"),)),
            sg.Button("Load Script")
        ],
        [sg.HorizontalSeparator()],

        # Static Analysis
        [sg.Text("Static Analysis:")],
        [sg.Button("Run Analysis")],
        [sg.Multiline(key="-ANALYSIS_OUT-", size=(80, 10))],
        [sg.HorizontalSeparator()],

        # Optimization Strategies
        [sg.Text("Optimization Strategies:")],
        [
            sg.Checkbox("Cython Memoryviews", key="-CYTHON-"),
            sg.Checkbox("Numba JIT", key="-NUMBA-"),
            sg.Checkbox("Preallocate Buffers", key="-PREALLOC-"),
            sg.Checkbox("Optimize Layout", key="-LAYOUT-"),
        ],
        [sg.Button("Build Optimization Plan")],
        [sg.Multiline(key="-PLAN_OUT-", size=(80, 10))],
        [sg.HorizontalSeparator()],

        # Code Generation
        [sg.Text("Code Generation:")],
        [sg.Button("Generate Code")],
        [sg.Multiline(key="-CODEGEN_OUT-", size=(80, 10))],
        [sg.HorizontalSeparator()],

        # Execution
        [sg.Text("Execution:")],
        [
            sg.Button("Run Baseline"),
            sg.Button("Run Optimized")
        ],
        [sg.Multiline(key="-EXEC_OUT-", size=(80, 10))],
        [sg.HorizontalSeparator()],

        # Plots
        [sg.Text("Plots:")],
        [sg.Button("Generate Plots")],
        [sg.Image(key="-PLOT_IMG-", size=(600, 300))],
    ]


# ============================================================
# GUI Application
# ============================================================

class MemAllocApp:
    def __init__(self):
        print(">>> ACTIVE APP.PY:", __file__)  # DEBUG

        theme = ThemeLoader(THEME_PATH).load()
        apply_theme(theme)

        self.controller = MemAllocController(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            db_path=DB_PATH,
            plots_dir=PLOTS_DIR,
        )

        layout = [
            [
                sg.Column(
                    build_main_layout(),
                    scrollable=True,
                    expand_x=True,
                    expand_y=True
                )
            ]
        ]

        self.window = sg.Window(
            "MemAlloc Optimizer",
            layout,
            finalize=True,
            resizable=True,
        )

    # --------------------------------------------------------
    # Main Event Loop
    # --------------------------------------------------------

    def run(self):
        while True:
            event, values = self.window.read()

            if event == sg.WIN_CLOSED:
                break

            if event == "Load Script":
                self._handle_load_script(values)

            elif event == "Run Analysis":
                self._handle_analysis()

            elif event == "Build Optimization Plan":
                self._handle_plan(values)

            elif event == "Generate Code":
                self._handle_codegen()

            elif event == "Run Baseline":
                self._handle_run_baseline()

            elif event == "Run Optimized":
                self._handle_run_optimized()

            elif event == "Generate Plots":
                self._handle_plots()

        self.window.close()

    # --------------------------------------------------------
    # Handlers
    # --------------------------------------------------------

    def _handle_load_script(self, values):
        path = Path(values["-SCRIPT_PATH-"])
        result = self.controller.load_script(path)

        vm = ScriptLoadVM(
            path=result["path"],
            hash=result["hash"],
            imports=result["imports"],
            entry_points=result["entry_points"],
            cached=result["cached"],
        )

        out = (
            f"Loaded script: {vm.path}\n"
            f"Hash: {vm.hash}\n"
            f"Imports: {vm.imports}\n"
            f"Entry Points: {vm.entry_points}\n"
            f"Cached: {vm.cached}\n"
        )

        self.window["-ANALYSIS_OUT-"].update(out)

    def _handle_analysis(self):
        result = self.controller.run_analysis()

        if "error" in result:
            self.window["-ANALYSIS_OUT-"].update(result["error"])
            return

        vm = AnalysisVM(
            hotspots=result["hotspots"],
            tips=result["tips"],
        )

        out = "Hotspots:\n"
        for h in vm.hotspots:
            out += f"- Line {h['line']}: {h['description']}\n"

        out += "\nMemory Tips:\n"
        for tip in vm.tips:
            out += f"- {tip}\n"

        self.window["-ANALYSIS_OUT-"].update(out)

    def _handle_plan(self, values):
        user_selection = {
            "cython_memoryviews": values["-CYTHON-"],
            "numba_jit": values["-NUMBA-"],
            "preallocate_buffers": values["-PREALLOC-"],
            "optimize_layout": values["-LAYOUT-"],
        }

        result = self.controller.build_plan(user_selection)

        if "error" in result:
            self.window["-PLAN_OUT-"].update(result["error"])
            return

        vm = OptimizationPlanVM(
            strategies=result["strategies"],
            notes=result["notes"],
        )

        out = "Optimization Plan:\n"
        for s in vm.strategies:
            out += f"- {s['name']}: {'enabled' if s['enabled'] else 'disabled'}\n"

        out += "\nNotes:\n"
        for n in vm.notes:
            out += f"- {n}\n"

        self.window["-PLAN_OUT-"].update(out)

    def _handle_codegen(self):
        result = self.controller.generate_code()

        if "error" in result:
            self.window["-CODEGEN_OUT-"].update(result["error"])
            return

        vm = CodegenVM(
            python_generated=result["python_generated"],
            cython_generated=result["cython_generated"],
            notes=result["notes"],
        )

        out = "Code Generation:\n"
        for n in vm.notes:
            out += f"- {n}\n"

        out += f"\nPython generated: {vm.python_generated}\n"
        out += f"Cython generated: {vm.cython_generated}\n"

        self.window["-CODEGEN_OUT-"].update(out)

    def _handle_run_baseline(self):
        result = self.controller.run_baseline()

        if "error" in result:
            self.window["-EXEC_OUT-"].update(result["error"])
            return

        vm = ExecutionVM(
            success=result["success"],
            runtime=result["runtime"],
            memory=result["memory"],
            stdout=result["stdout"],
            stderr=result["stderr"],
        )

        out = (
            f"Baseline Execution:\n"
            f"Runtime: {vm.runtime:.4f}s\n"
            f"Peak Memory: {vm.memory:.2f} MB\n"
            f"Success: {vm.success}\n"
            f"Stdout:\n{vm.stdout}\n"
            f"Stderr:\n{vm.stderr}\n"
        )

        self.window["-EXEC_OUT-"].update(out)

    def _handle_run_optimized(self):
        result = self.controller.run_optimized()

        if "error" in result:
            self.window["-EXEC_OUT-"].update(result["error"])
            return

        vm = ExecutionVM(
            success=result["success"],
            runtime=result["runtime"],
            memory=result["memory"],
            stdout=result["stdout"],
            stderr=result["stderr"],
        )

        out = (
            f"Optimized Execution:\n"
            f"Runtime: {vm.runtime:.4f}s\n"
            f"Peak Memory: {vm.memory:.2f} MB\n"
            f"Success: {vm.success}\n"
            f"Stdout:\n{vm.stdout}\n"
            f"Stderr:\n{vm.stderr}\n"
        )

        self.window["-EXEC_OUT-"].update(out)

    def _handle_plots(self):
        result = self.controller.generate_plots()

        vm = PlotsVM(
            memory_plot=result["memory_plot"],
            runtime_plot=result["runtime_plot"],
            speedup_plot=result["speedup_plot"],
        )

        if vm.memory_plot:
            self.window["-PLOT_IMG-"].update(filename=vm.memory_plot)


# ============================================================
# Entry Point
# ============================================================

def main():
    app = MemAllocApp()
    app.run()


if __name__ == "__main__":
    main()
