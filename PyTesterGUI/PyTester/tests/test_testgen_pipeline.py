"""
Tests for testgen/ modules.

These tests validate:
- smoke test generation
- type test generation
- boundary test generation
- property test generation
- docstring test generation
- template rendering

They intentionally avoid executing user code.
They only verify that generators produce syntactically valid pytest code.
"""

from pathlib import Path

import testgen.smoke_generator as smoke_generator
import testgen.type_tests_generator as type_tests_generator
import testgen.boundary_tests_generator as boundary_tests_generator
import testgen.property_tests_generator as property_tests_generator
import testgen.docstring_tests_generator as docstring_tests_generator
import testgen.template_renderer as template_renderer


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _dummy_schema():
    return {
        "add": {
            "kind": "function",
            "args": {"a": "int", "b": "float"},
            "return": "float",
            "docstring": "Returns float",
            "intent": "computation",
        },
        "Util.process": {
            "kind": "method",
            "args": {"x": "str"},
            "return": "str",
            "docstring": "Output is str",
            "intent": "transformation",
        },
    }


def _dummy_settings():
    return {
        "testgen": {
            "smoke": {"enable_runtime_checks": False},
            "type": {"enable_runtime_checks": False},
            "boundary": {"enable_runtime_checks": False},
            "property": {"enable_runtime_checks": False},
            "docstring": {"enable_runtime_checks": False},
            "renderer": {
                "indent_spaces": 4,
                "output_dir": "workspace/generated_tests"
            },
        }
    }


# ------------------------------------------------------------
# smoke_generator
# ------------------------------------------------------------
def test_smoke_generator_produces_pytest_code(tmp_path):
    gen = smoke_generator.SmokeGenerator(_dummy_settings())
    schema = _dummy_schema()

    content = gen.generate(Path("dummy.py"), schema)

    assert "def test_smoke_add" in content
    assert "def test_smoke_Util_process" in content
    assert "pytest" in content


# ------------------------------------------------------------
# type_tests_generator
# ------------------------------------------------------------
def test_type_tests_generator_produces_pytest_code(tmp_path):
    gen = type_tests_generator.TypeTestsGenerator(_dummy_settings())
    schema = _dummy_schema()

    content = gen.generate(Path("dummy.py"), schema)

    assert "test_type_add" in content
    assert "assert callable(func)" in content


# ------------------------------------------------------------
# boundary_tests_generator
# ------------------------------------------------------------
def test_boundary_tests_generator_produces_pytest_code(tmp_path):
    gen = boundary_tests_generator.BoundaryTestsGenerator(_dummy_settings())
    schema = _dummy_schema()

    content = gen.generate(Path("dummy.py"), schema)

    assert "test_boundary_add" in content
    assert "assert True" in content


# ------------------------------------------------------------
# property_tests_generator
# ------------------------------------------------------------
def test_property_tests_generator_produces_pytest_code(tmp_path):
    gen = property_tests_generator.PropertyTestsGenerator(_dummy_settings())
    schema = _dummy_schema()

    content = gen.generate(Path("dummy.py"), schema)

    assert "test_property_add" in content
    assert "deterministic output" in content


# ------------------------------------------------------------
# docstring_tests_generator
# ------------------------------------------------------------
def test_docstring_tests_generator_produces_pytest_code(tmp_path):
    gen = docstring_tests_generator.DocstringTestsGenerator(_dummy_settings())
    schema = _dummy_schema()

    content = gen.generate(Path("dummy.py"), schema)

    assert "test_docstring_add" in content
    assert "docstring claim" in content


# ------------------------------------------------------------
# template_renderer
# ------------------------------------------------------------
def test_template_renderer_writes_file(tmp_path):
    settings = _dummy_settings()
    settings["testgen"]["renderer"]["output_dir"] = str(tmp_path)

    renderer = template_renderer.TemplateRenderer(settings)

    output = renderer.render("test_file.py", "content")
    assert output.exists()
    assert output.read_text() == "content"


def test_template_renderer_wrap_test_case():
    renderer = template_renderer.TemplateRenderer(_dummy_settings())

    wrapped = renderer.wrap_test_case("test_x", "assert True")
    assert "def test_x" in wrapped
    assert "assert True" in wrapped
