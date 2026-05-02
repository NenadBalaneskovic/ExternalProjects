# src/pynonym/__init__.py

"""
pynonym — Multilingual text and table anonymization toolkit.

Exports:
- High-level APIs:
    anonymize_text
    anonymize_dataframe

- Configuration:
    PynonymConfig
    TableAnonymizationConfig

- Anonymizer classes:
    TextAnonymizer
    TableAnonymizer
"""

from .config import PynonymConfig
from .text import TextAnonymizer, anonymize_text
from .tables import TableAnonymizer, TableAnonymizationConfig, anonymize_dataframe

__all__ = [
    "PynonymConfig",
    "TableAnonymizationConfig",
    "TextAnonymizer",
    "TableAnonymizer",
    "anonymize_text",
    "anonymize_dataframe",
]
