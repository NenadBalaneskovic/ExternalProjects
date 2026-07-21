"""
Utils

This module provides small, pure utility functions used across PyTester:
- safe path handling
- dictionary merging
- list flattening
- simple formatting helpers

It is intentionally minimal and deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ------------------------------------------------------------
# Path utilities
# ------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    """
    Ensure that a directory exists.

    Parameters
    ----------
    path : Path
        Directory path to create if missing.
    """
    path.mkdir(parents=True, exist_ok=True)


def read_text_safe(path: Path) -> Optional[str]:
    """
    Safely read a text file.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    Optional[str]
        File contents or None if reading fails.
    """
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def write_text_safe(path: Path, content: str) -> bool:
    """
    Safely write text to a file.

    Parameters
    ----------
    path : Path
        File to write.

    content : str
        Text content.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception:
        return False


# ------------------------------------------------------------
# Dictionary utilities
# ------------------------------------------------------------
def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries shallowly.

    Parameters
    ----------
    a : dict
        First dictionary.

    b : dict
        Second dictionary.

    Returns
    -------
    dict
        Combined dictionary.
    """
    merged = dict(a)
    merged.update(b)
    return merged


def deep_merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Parameters
    ----------
    a : dict
        First dictionary.

    b : dict
        Second dictionary.

    Returns
    -------
    dict
        Deeply merged dictionary.
    """
    result = dict(a)

    for key, value in b.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


# ------------------------------------------------------------
# List utilities
# ------------------------------------------------------------
def flatten(items: Iterable[Iterable[Any]]) -> List[Any]:
    """
    Flatten a list of lists.

    Parameters
    ----------
    items : iterable
        Iterable of iterables.

    Returns
    -------
    list
        Flattened list.
    """
    return [x for sub in items for x in sub]


# ------------------------------------------------------------
# Formatting utilities
# ------------------------------------------------------------
def indent(text: str, spaces: int = 4) -> str:
    """
    Indent each line of a text block.

    Parameters
    ----------
    text : str
        Text to indent.

    spaces : int
        Number of spaces.

    Returns
    -------
    str
        Indented text.
    """
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


def safe_repr(obj: Any) -> str:
    """
    Safe representation of an object.

    Parameters
    ----------
    obj : Any
        Object to represent.

    Returns
    -------
    str
        String representation, falling back to <unrepr> if needed.
    """
    try:
        return repr(obj)
    except Exception:
        return "<unrepr>"
