"""
Tests for core/ modules.

These tests validate:
- basic importability
- deterministic behavior of utility functions
- correct AST extraction
- correct docstring extraction
- correct annotation extraction
- correct structure registry behavior

They intentionally avoid executing user code.
"""

from pathlib import Path
import types

import core.input_loader as input_loader
import core.syntax_checker as syntax_checker
import core.safe_import as safe_import
import core.ast_inspector as ast_inspector
import core.docstring_extractor as docstring_extractor
import core.annotation_extractor as annotation_extractor
import core.structure_registry as structure_registry
import core.utils as utils


# ------------------------------------------------------------
# input_loader
# ------------------------------------------------------------
def test_input_loader_reads_file(tmp_path):
    p = tmp_path / "sample.py"
    p.write_text("x = 1\n")

    content = input_loader.load_file(p)
    assert "x = 1" in content


# ------------------------------------------------------------
# syntax_checker
# ------------------------------------------------------------
def test_syntax_checker_valid_code():
    code = "a = 1\nb = a + 2"
    result = syntax_checker.check_syntax(code)
    assert result["valid"] is True
    assert result["error"] is None


def test_syntax_checker_invalid_code():
    code = "a ="
    result = syntax_checker.check_syntax(code)
    assert result["valid"] is False
    assert isinstance(result["error"], SyntaxError)


# ------------------------------------------------------------
# safe_import
# ------------------------------------------------------------
def test_safe_import_basic(tmp_path):
    p = tmp_path / "mod.py"
    p.write_text("x = 42")

    module = safe_import.safe_import(p)
    assert isinstance(module, types.ModuleType)
    assert hasattr(module, "x")
    assert module.x == 42


# ------------------------------------------------------------
# ast_inspector
# ------------------------------------------------------------
def test_ast_inspector_extracts_functions():
    code = """
def foo(a, b):
    return a + b

class Bar:
    def baz(self):
        return 1
"""
    tree = ast_inspector.parse_ast(code)
    symbols = ast_inspector.extract_symbols(tree)

    assert "foo" in symbols
    assert "Bar.baz" in symbols


# ------------------------------------------------------------
# docstring_extractor
# ------------------------------------------------------------
def test_docstring_extractor_function():
    code = '''
def foo():
    """This is a test docstring."""
    return 1
'''
    tree = ast_inspector.parse_ast(code)
    symbols = ast_inspector.extract_symbols(tree)

    docs = docstring_extractor.extract_docstrings(tree, symbols)
    assert docs["foo"] == "This is a test docstring."


# ------------------------------------------------------------
# annotation_extractor
# ------------------------------------------------------------
def test_annotation_extractor_function_annotations():
    code = """
def add(a: int, b: float) -> float:
    return a + b
"""
    tree = ast_inspector.parse_ast(code)
    symbols = ast_inspector.extract_symbols(tree)

    ann = annotation_extractor.extract_annotations(tree, symbols)
    assert ann["add"]["args"]["a"] == "int"
    assert ann["add"]["args"]["b"] == "float"
    assert ann["add"]["return"] == "float"


# ------------------------------------------------------------
# structure_registry
# ------------------------------------------------------------
def test_structure_registry_register_and_get():
    reg = structure_registry.StructureRegistry()
    reg.register("foo", {"kind": "function"})
    reg.register("Bar.baz", {"kind": "method"})

    assert reg.get("foo")["kind"] == "function"
    assert reg.get("Bar.baz")["kind"] == "method"


# ------------------------------------------------------------
# utils
# ------------------------------------------------------------
def test_utils_indent():
    text = "hello"
    indented = utils.indent(text, 4)
    assert indented == "    hello"


def test_utils_ensure_dir(tmp_path):
    d = tmp_path / "newdir"
    utils.ensure_dir(d)
    assert d.exists()


def test_utils_write_text_safe(tmp_path):
    p = tmp_path / "file.txt"
    utils.write_text_safe(p, "content")
    assert p.read_text() == "content"
