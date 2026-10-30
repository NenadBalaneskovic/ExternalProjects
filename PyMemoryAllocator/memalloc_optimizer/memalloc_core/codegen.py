"""
codegen.py

Responsible for:
- Generating optimized Python and Cython code based on the OptimizationPlan
- Producing Cython modules (.pyx) when required
- Producing Numba-decorated Python modules when required
- Applying preallocation and memory-layout transformations
- Returning structured results for GUI preview and execution sandbox
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from .optimization_engine import OptimizationPlan
import ast
import astor


# ============================================================
# Data structures
# ============================================================

@dataclass
class CodegenResult:
    """Result of code generation."""
    optimized_python: Optional[str]
    optimized_cython: Optional[str]
    output_dir: Path
    notes: List[str]


# ============================================================
# Code Generator
# ============================================================

class CodeGenerator:
    """
    Generates optimized Python and Cython code based on the OptimizationPlan.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def generate(self, plan: OptimizationPlan, ast_tree: ast.AST) -> CodegenResult:
        optimized_python = self._generate_python_code(plan, ast_tree)
        notes = ["Generated optimized Python module."]

        optimized_cython = None
        if plan.cython_required:
            optimized_cython = self._generate_cython_code(plan, ast_tree)
            notes.append("Generated Cython module with memoryviews and arena allocator.")

        return CodegenResult(
            optimized_python=optimized_python,
            optimized_cython=optimized_cython,
            output_dir=self.output_dir,
            notes=notes
        )

    # --------------------------------------------------------
    # Remove GUI imports (CRITICAL FIX)
    # --------------------------------------------------------

    def _remove_gui_imports(self, tree: ast.AST) -> ast.AST:
        class ImportStripper(ast.NodeTransformer):
            def visit_Import(self, node):
                node.names = [
                    n for n in node.names
                    if not n.name.startswith("memalloc_gui")
                ]
                return node

            def visit_ImportFrom(self, node):
                if node.module and node.module.startswith("memalloc_gui"):
                    return None
                return node

        return ImportStripper().visit(tree)

    # --------------------------------------------------------
    # Remove main() wrapper (CRITICAL FIX)
    # --------------------------------------------------------

    def _remove_main_wrapper(self, tree: ast.AST) -> ast.AST:
        class MainStripper(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == "main":
                    return None
                return node

            def visit_If(self, node):
                # Remove: if __name__ == "__main__":
                if isinstance(node.test, ast.Compare):
                    return None
                return node

        return MainStripper().visit(tree)

    # --------------------------------------------------------
    # Python code generation
    # --------------------------------------------------------

    def _generate_python_code(self, plan: OptimizationPlan, ast_tree: ast.AST) -> str:
        tree = ast_tree

        # ⭐ Remove GUI imports
        tree = self._remove_gui_imports(tree)

        # ⭐ Remove main() wrapper
        tree = self._remove_main_wrapper(tree)

        # Ensure numpy is imported
        numpy_import = ast.Import(names=[ast.alias(name="numpy", asname="np")])
        tree.body.insert(0, numpy_import)

        # Apply Numba JIT if required
        if plan.numba_required:
            tree = self._apply_numba_jit(tree)

        # Apply preallocation strategies
        if plan.preallocation_required:
            tree = self._apply_preallocation(tree)

        # Apply layout optimization hints
        if plan.layout_opt_required:
            tree = self._apply_layout_hints(tree)

        # Convert AST back to Python code
        optimized_code = astor.to_source(tree)

        # Write to file
        out_file = self.output_dir / "optimized_module.py"
        out_file.write_text(optimized_code)

        return optimized_code

    # --------------------------------------------------------
    # Cython code generation
    # --------------------------------------------------------

    def _generate_cython_code(self, plan: OptimizationPlan, ast_tree: ast.AST) -> str:
        cython_template = """
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

# Arena allocator for memoryviews
cdef class Arena:
    cdef double[:] buffer
    cdef Py_ssize_t size
    cdef Py_ssize_t offset

    def __init__(self, int n):
        self.buffer = cnp.zeros(n, dtype=cnp.float64)
        self.size = n
        self.offset = 0

    cdef double[:] alloc(self, int n):
        if self.offset + n > self.size:
            raise MemoryError("Arena overflow")
        view = self.buffer[self.offset:self.offset+n]
        self.offset += n
        return view

# Example optimized function (placeholder)
def optimized_kernel(double[:] data):
    cdef Py_ssize_t i
    cdef double acc = 0
    for i in range(data.shape[0]):
        acc += data[i] * 0.5
    return acc
"""

        out_file = self.output_dir / "optimized_module.pyx"
        out_file.write_text(cython_template)

        return cython_template

    # --------------------------------------------------------
    # Python-level transformations
    # --------------------------------------------------------

    def _apply_numba_jit(self, tree: ast.AST) -> ast.AST:
        numba_import = ast.Import(names=[ast.alias(name="numba", asname=None)])
        tree.body.insert(1, numba_import)  # after numpy import

        class NumbaTransformer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                decorator = ast.Attribute(
                    value=ast.Name(id="numba", ctx=ast.Load()),
                    attr="njit",
                    ctx=ast.Load()
                )
                node.decorator_list.append(decorator)
                return node

        return NumbaTransformer().visit(tree)

    def _apply_preallocation(self, tree: ast.AST) -> ast.AST:
        class PreallocTransformer(ast.NodeTransformer):
            def visit_For(self, node):
                new_body = []
                hoisted = []

                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                        hoisted.append(stmt)
                    else:
                        new_body.append(stmt)

                node.body = new_body
                return hoisted + [node]

        return PreallocTransformer().visit(tree)

    def _apply_layout_hints(self, tree: ast.AST) -> ast.AST:
        class LayoutTransformer(ast.NodeTransformer):
            def _get_func_name(self, node):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            return f"{node.func.value.id}.{node.func.attr}"
                    elif isinstance(node.func, ast.Name):
                        return node.func.id
                return None

            def visit_Assign(self, node):
                if isinstance(node.value, ast.Call):
                    func_name = self._get_func_name(node.value)
                    if func_name in ("np.zeros", "np.ones", "np.random.rand"):
                        node.value = ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="np", ctx=ast.Load()),
                                attr="ascontiguousarray",
                                ctx=ast.Load()
                            ),
                            args=[node.value],
                            keywords=[]
                        )
                return self.generic_visit(node)

        return LayoutTransformer().visit(tree)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _get_func_name(self, call: ast.Call) -> str:
        if isinstance(call.func, ast.Attribute):
            return f"{call.func.value.id}.{call.func.attr}"
        if isinstance(call.func, ast.Name):
            return call.func.id
        return "unknown"
