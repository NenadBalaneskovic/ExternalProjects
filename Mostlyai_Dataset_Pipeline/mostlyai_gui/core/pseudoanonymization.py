"""
pseudonymization.py

Backend pseudonymization engine for the Data Privacy Workbench GUI.

This module provides a deterministic pseudonymization pipeline that supports
the actions defined in the GUI:

    - pseudonymize: Replace values with deterministic pseudonyms
    - hash: SHA-256 hash of the value
    - none: Leave unchanged

Additionally, it supports:
    - Optional mapping table persistence (for reversible pseudonymization)
    - Deterministic pseudonym generation using a seeded RNG

Author: Nenad
Date: May 2026
"""

import hashlib
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path


class PseudonymizationPipeline:
    """
    Pseudonymization pipeline.

    Parameters
    ----------
    rules : dict
        Mapping of column -> action ("pseudonymize", "hash", "none").
    store_mapping : bool
        Whether to save the pseudonymization mapping table.
    mapping_path : str
        Directory where mapping tables should be stored.
    logger : logging.Logger
        Logger instance for reporting progress.
    """

    def __init__(self, rules, store_mapping=False, mapping_path="output/mappings", logger=None):
        self.rules = rules
        self.store_mapping = store_mapping
        self.mapping_path = Path(mapping_path)
        self.logger = logger

        # Ensure mapping directory exists
        if self.store_mapping:
            self.mapping_path.mkdir(parents=True, exist_ok=True)

        # Global mapping dictionary: {column: {original: pseudonym}}
        self.mapping = {}

        # Deterministic RNG for pseudonym generation
        self.rng = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply pseudonymization rules to the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.

        Returns
        -------
        pandas.DataFrame
            Pseudonymized dataset.
        """
        if self.logger:
            self.logger.info("Starting pseudonymization pipeline...")

        df = df.copy()

        for column, action in self.rules.items():
            if column not in df.columns:
                self._log(f"Column '{column}' not found in dataset.")
                continue

            if action == "pseudonymize":
                df[column] = self._pseudonymize_column(df[column], column)
                self._log(f"Pseudonymized column: {column}")

            elif action == "hash":
                df[column] = df[column].astype(str).apply(self._hash_value)
                self._log(f"Hashed column: {column}")

            elif action == "none":
                self._log(f"Left unchanged: {column}")

            else:
                self._log(f"Unknown action '{action}' for column '{column}'")

        # Save mapping table if enabled
        if self.store_mapping:
            self._save_mapping()

        self._log("Pseudonymization pipeline completed.")
        return df

    # ------------------------------------------------------------------
    # Pseudonymization logic
    # ------------------------------------------------------------------

    def _pseudonymize_column(self, series: pd.Series, column: str) -> pd.Series:
        """
        Pseudonymize a column deterministically.

        Parameters
        ----------
        series : pandas.Series
            Column to pseudonymize.
        column : str
            Column name.

        Returns
        -------
        pandas.Series
            Pseudonymized column.
        """
        unique_values = series.dropna().unique()

        # Initialize mapping for this column
        self.mapping[column] = {}

        for value in unique_values:
            pseudo = self._generate_pseudonym()
            self.mapping[column][str(value)] = pseudo

        # Apply mapping
        return series.astype(str).map(self.mapping[column]).fillna(series)

    def _generate_pseudonym(self) -> str:
        """
        Generate a deterministic pseudonym using a seeded RNG.

        Returns
        -------
        str
            Pseudonym string.
        """
        # Example pseudonym format: "PSE-483920"
        number = self.rng.integers(100000, 999999)
        return f"PSE-{number}"

    # ------------------------------------------------------------------
    # Hashing logic
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_value(value: str) -> str:
        """Return SHA-256 hash of the value."""
        if value is None:
            return None
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Mapping persistence
    # ------------------------------------------------------------------

    def _save_mapping(self):
        """Save mapping tables to JSON files."""
        for column, mapping_dict in self.mapping.items():
            path = self.mapping_path / f"{column}_mapping.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(mapping_dict, f, indent=4, ensure_ascii=False)
            self._log(f"Saved mapping table: {path}")

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(self, message: str):
        if self.logger:
            self.logger.info(f"Pseudonymization: {message}")
