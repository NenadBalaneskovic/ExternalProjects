"""
utils/file_helpers.py

Responsible for:
- Safe, reusable filesystem utilities used across PyDependencyInspector.
- Reading/writing text files.
- Normalizing paths.
- Ensuring directories exist.
- Copying files (for data/binary inclusion in PyInstaller builds).
- Providing deterministic, GUI-agnostic behavior.

This module is intentionally:
- Pure Python.
- Safe in offline environments.
- Free of GUI dependencies.
"""

from __future__ import annotations

import os
import shutil
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def ensure_directory(path: str) -> None:
    """
    Ensure that a directory exists. Create it if necessary.

    :param path: Directory path
    """
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as exc:
        logger.exception("Failed to create directory '%s': %s", path, exc)
        raise


def normalize_path(path: str) -> str:
    """
    Normalize a filesystem path to an absolute, platform-correct form.

    :param path: Input path
    :return: Normalized absolute path
    """
    return os.path.abspath(os.path.expanduser(path))


def file_exists(path: str) -> bool:
    """
    Check whether a file exists.

    :param path: File path
    :return: True if file exists
    """
    return os.path.isfile(path)


def directory_exists(path: str) -> bool:
    """
    Check whether a directory exists.

    :param path: Directory path
    :return: True if directory exists
    """
    return os.path.isdir(path)


# ---------------------------------------------------------------------------
# File read/write utilities
# ---------------------------------------------------------------------------

def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """
    Read a text file safely.

    :param path: File path
    :param encoding: Text encoding
    :return: File content as string
    """
    try:
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as exc:
        logger.exception("Failed to read file '%s': %s", path, exc)
        raise


def write_text_file(path: str, content: str, encoding: str = "utf-8") -> None:
    """
    Write text content to a file safely.

    :param path: File path
    :param content: Text content
    :param encoding: Text encoding
    """
    try:
        ensure_directory(os.path.dirname(path))
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
    except Exception as exc:
        logger.exception("Failed to write file '%s': %s", path, exc)
        raise


# ---------------------------------------------------------------------------
# File copying utilities
# ---------------------------------------------------------------------------

def copy_file(src: str, dest: str) -> None:
    """
    Copy a file from src to dest, creating directories if needed.

    :param src: Source file path
    :param dest: Destination file path
    """
    try:
        ensure_directory(os.path.dirname(dest))
        shutil.copy2(src, dest)
    except Exception as exc:
        logger.exception("Failed to copy file '%s' -> '%s': %s", src, dest, exc)
        raise


def safe_remove(path: str) -> None:
    """
    Remove a file if it exists. Ignore if missing.

    :param path: File path
    """
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as exc:
        logger.exception("Failed to remove file '%s': %s", path, exc)
        raise


# ---------------------------------------------------------------------------
# Temporary file helpers
# ---------------------------------------------------------------------------

def create_temp_file(prefix: str = "tmp_", suffix: str = ".txt") -> str:
    """
    Create a temporary file path (not created on disk).

    :param prefix: Filename prefix
    :param suffix: Filename suffix
    :return: Absolute path to a non-existing temp file
    """
    import tempfile
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)  # We only want the path, not an open file descriptor
    return path


def create_temp_directory(prefix: str = "tmpdir_") -> str:
    """
    Create a temporary directory.

    :param prefix: Directory prefix
    :return: Absolute path to the created directory
    """
    import tempfile
    return tempfile.mkdtemp(prefix=prefix)
