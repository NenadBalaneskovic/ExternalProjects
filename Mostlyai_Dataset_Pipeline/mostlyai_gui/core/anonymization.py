"""
anonymization.py

Backend anonymization engine for the Data Privacy Workbench GUI.

This module provides a simple, deterministic anonymization pipeline
that supports the actions defined in the GUI:

    - mask: Replace characters with '*'
    - generalize: Replace values with coarse categories
    - remove: Drop the column entirely
    - hash: SHA-256 hash of the value
    - none: Leave unchanged

This module is intentionally lightweight and extensible. You can later
replace or augment it with:
    - spaCy NER anonymization
    - Faker-based replacements
    - Pynonym integration
    - k-anonymity / l-diversity checks

Author: Nenad
Date: May 2026
"""

import hashlib
import pandas as pd


class AnonymizationPipeline:
    """
    Anonymization pipeline.

    Parameters
    ----------
    rules : dict
        Mapping of column -> action ("mask", "generalize", "remove", "hash", "none").
    logger : logging.Logger
        Logger instance for reporting progress.
    """

    def __init__(self, rules: dict, logger=None):
        self.rules = rules
        self.logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply anonymization rules to the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.

        Returns
        -------
        pandas.DataFrame
            Anonymized dataset.
        """
        if self.logger:
            self.logger.info("Starting anonymization pipeline...")

        df = df.copy()

        for column, action in self.rules.items():
            if column not in df.columns:
                if self.logger:
                    self.logger.warning(f"Column '{column}' not found in dataset.")
                continue

            if action == "mask":
                df[column] = df[column].astype(str).apply(self._mask_value)
                self._log(f"Masked column: {column}")

            elif action == "generalize":
                df[column] = df[column].apply(self._generalize_value)
                self._log(f"Generalized column: {column}")

            elif action == "remove":
                df = df.drop(columns=[column])
                self._log(f"Removed column: {column}")

            elif action == "hash":
                df[column] = df[column].astype(str).apply(self._hash_value)
                self._log(f"Hashed column: {column}")

            elif action == "none":
                self._log(f"Left unchanged: {column}")

            else:
                self._log(f"Unknown action '{action}' for column '{column}'")

        self._log("Anonymization pipeline completed.")
        return df

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_value(value: str) -> str:
        """Replace characters with '*' while preserving length."""
        if value is None:
            return None
        return "*" * len(str(value))

    @staticmethod
    def _generalize_value(value):
        """
        Coarse generalization:
            - Numbers → buckets
            - Strings → first letter + '***'
        """
        if pd.isna(value):
            return value

        # Numeric generalization
        if isinstance(value, (int, float)):
            if value < 18:
                return "<18"
            elif value < 30:
                return "18-29"
            elif value < 50:
                return "30-49"
            elif value < 70:
                return "50-69"
            else:
                return "70+"

        # String generalization
        value = str(value)
        if len(value) == 0:
            return value
        return value[0] + "***"

    @staticmethod
    def _hash_value(value: str) -> str:
        """Return SHA-256 hash of the value."""
        if value is None:
            return None
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(self, message: str):
        if self.logger:
            self.logger.info(f"Anonymization: {message}")
