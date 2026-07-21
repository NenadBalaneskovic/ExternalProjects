"""
Tests for inference/ modules.

These tests validate:
- static analysis (AST-based)
- semantic analysis (name-based heuristics)
- dynamic probing (safe runtime calls)
- type fusion (combining multiple inference signals)
- schema builder (canonical schema assembly)

They intentionally avoid executing arbitrary user code.
"""

from pathlib import Path
import types

import inference.static_analysis as static_analysis
import inference.semantic_analysis as semantic_analysis
import inference.dynamic_probe as dynamic_probe
import inference.type_fusion as type_fusion
import inference.schema_builder as schema_builder

import core.ast_inspector as ast_inspector
import core.structure_registry as structure_registry


# ------------------------------------------------------------
# static_analysis
# ------------------------------------------------------------
def test_static_analysis_extracts_types():
    code = """
def add(a: int, b: float) -> float:
    return a + b
"""
    tree = ast_inspector.parse_ast(code)
    symbols = ast_inspector.extract_symbols(tree)

    result = static_analysis.infer_static_types(tree, symbols)

    assert result["add"]["args"]["a"] == "int"
    assert result["add"]["args"]["b"] == "float"
    assert result["add"]["return"] == "float"


def test_static_analysis_handles_missing_annotations():
    code = """
def foo(x):
    return x
"""
    tree = ast_inspector.parse_ast(code)
    symbols = ast_inspector.extract_symbols(tree)

    result = static_analysis.infer_static_types(tree, symbols)

    assert result["foo"]["args"]["x"] is None
    assert result["foo"]["return"] is None


# ------------------------------------------------------------
# semantic_analysis
# ------------------------------------------------------------
def test_semantic_analysis_name_based_inference():
    info = {
        "args": {"count": None, "flag": None},
        "return": None,
    }

    result = semantic_analysis.infer_semantic_types("process_count", info)

    assert result["args"]["count"] == "int"
    assert result["args"]["flag"] == "bool"


def test_semantic_analysis_no_guess_for_unknown_names():
    info = {"args": {"x": None}, "return": None}
    result = semantic_analysis.infer_semantic_types("mystery", info)

    assert result["args"]["x"] is None


# ------------------------------------------------------------
# dynamic_probe
# ------------------------------------------------------------
def test_dynamic_probe_safe_execution(tmp_path):
    p = tmp_path / "mod.py"
    p.write_text("""
def double(x):
    return x * 2
""")

    module = dynamic_probe.safe_import_for_probe(p)
    result = dynamic_probe.probe_return_type(module.double, [1])

    assert result == "int"


def test_dynamic_probe_handles_exceptions(tmp_path):
    p = tmp_path / "mod.py"
    p.write_text("""
def explode(x):
    raise ValueError("boom")
""")

    module = dynamic_probe.safe_import_for_probe(p)
    result = dynamic_probe.probe_return_type(module.explode, [1])

    assert result is None


# ------------------------------------------------------------
# type_fusion
# ------------------------------------------------------------
def test_type_fusion_combines_signals():
    static = {"args": {"a": "int"}, "return": "int"}
    semantic = {"args": {"a": None}, "return": None}
    dynamic = {"args": {"a": "int"}, "return": "int"}

    fused = type_fusion.fuse_types(static, semantic, dynamic)

    assert fused["args"]["a"] == "int"
    assert fused["return"] == "int"


def test_type_fusion_prefers_static_over_semantic():
    static = {"args": {"x": "float"}, "return": None}
    semantic = {"args": {"x": "int"}, "return": None}
    dynamic = {"args": {"x": None}, "return": None}

    fused = type_fusion.fuse_types(static, semantic, dynamic)

    assert fused["args"]["x"] == "float"


# ------------------------------------------------------------
# schema_builder
# ------------------------------------------------------------
def test_schema_builder_creates_canonical_schema():
    code = """
def add(a: int, b: float) -> float:
    return a + b
"""
    tree = ast_inspector.parse_ast(code)
    symbols = ast_inspector.extract_symbols(tree)

    reg = structure_registry.StructureRegistry()
    for name, info in symbols.items():
        reg.register(name, info)

    static = static_analysis.infer_static_types(tree, symbols)
    semantic = semantic_analysis.infer_semantic_types("add", static["add"])
    dynamic = {"args": {"a": "int", "b": "float"}, "return": "float"}

    builder = schema_builder.SchemaBuilder()
    schema = builder.build(reg, static, {"add": semantic}, {"add": dynamic})

    assert "add" in schema
    assert schema["add"]["args"]["a"] == "int"
    assert schema["add"]["args"]["b"] == "float"
    assert schema["add"]["return"] == "float"
    assert schema["add"]["kind"] == "function"
