# mlflow_local_runner/tests/test_validators.py
"""
test_validators.py – Tests für validators.py
"""

import pytest
from pathlib import Path

from utils.validators import (
    validate_file,
    validate_directory,
    validate_python_file,
    validate_csv_file,
    validate_uri,
)


# ---------------------------------------------------------
# validate_file
# ---------------------------------------------------------

def test_validate_file_existing(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello")
    assert validate_file(str(f)) is True


def test_validate_file_missing(tmp_path):
    f = tmp_path / "missing.txt"
    assert validate_file(str(f)) is False


def test_validate_file_none():
    assert validate_file(None) is False


# ---------------------------------------------------------
# validate_directory
# ---------------------------------------------------------

def test_validate_directory_existing(tmp_path):
    assert validate_directory(str(tmp_path)) is True


def test_validate_directory_missing(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert validate_directory(str(missing)) is False


def test_validate_directory_none():
    assert validate_directory(None) is False


# ---------------------------------------------------------
# validate_python_file
# ---------------------------------------------------------

def test_validate_python_file_valid(tmp_path):
    f = tmp_path / "script.py"
    f.write_text("print('hi')")
    assert validate_python_file(str(f)) is True


def test_validate_python_file_wrong_extension(tmp_path):
    f = tmp_path / "script.txt"
    f.write_text("print('hi')")
    assert validate_python_file(str(f)) is False


def test_validate_python_file_missing(tmp_path):
    f = tmp_path / "missing.py"
    assert validate_python_file(str(f)) is False


# ---------------------------------------------------------
# validate_csv_file
# ---------------------------------------------------------

def test_validate_csv_file_valid(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b,c\n1,2,3")
    assert validate_csv_file(str(f)) is True


def test_validate_csv_file_wrong_extension(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("a,b,c\n1,2,3")
    assert validate_csv_file(str(f)) is False


def test_validate_csv_file_missing(tmp_path):
    f = tmp_path / "missing.csv"
    assert validate_csv_file(str(f)) is False


# ---------------------------------------------------------
# validate_uri
# ---------------------------------------------------------

def test_validate_uri_valid():
    assert validate_uri("http://localhost:5000") is True
    assert validate_uri("https://example.com") is True


def test_validate_uri_invalid():
    assert validate_uri("localhost:5000") is False
    assert validate_uri("not a uri") is False
    assert validate_uri("") is False
    assert validate_uri(None) is False
