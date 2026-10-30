"""
cleaning.py

Backend cleaning engine for the Data Privacy Workbench GUI.

This module provides a simple, deterministic cleaning pipeline that supports
the options defined in the GUI:

    - Remove duplicates
    - Handle missing values
    - Normalize data types
    - Detect & fix outliers (IQR-based capping)

The goal is to provide a clean, reliable preprocessing step before
anonymization, pseudonymization, or synthetic data generation.

Author: Nenad
Date: May 2026
"""

import pandas as pd
import numpy as np


class CleaningPipeline:
    """
    Cleaning pipeline.

    Parameters
    ----------
    remove_duplicates : bool
        Whether to drop duplicate rows.
    handle_missing : bool
        Whether to fill missing values.
    normalize_types : bool
        Whether to normalize column data types.
    fix_outliers : bool
        Whether to cap outliers using IQR.
    logger : logging.Logger
        Logger instance for reporting progress.
    """

    def __init__(
        self,
        remove_duplicates=True,
        handle_missing=True,
        normalize_types=True,
        fix_outliers=True,
        logger=None,
    ):
        self.remove_duplicates = remove_duplicates
        self.handle_missing = handle_missing
        self.normalize_types = normalize_types
        self.fix_outliers = fix_outliers
        self.logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the cleaning pipeline.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.

        Returns
        -------
        pandas.DataFrame
            Cleaned dataset.
        """
        self._log("Starting cleaning pipeline...")
        df = df.copy()

        if self.remove_duplicates:
            df = self._remove_duplicates(df)

        if self.handle_missing:
            df = self._handle_missing(df)

        if self.normalize_types:
            df = self._normalize_types(df)

        if self.fix_outliers:
            df = self._fix_outliers(df)

        self._log("Cleaning pipeline completed.")
        return df

    # ------------------------------------------------------------------
    # Cleaning steps
    # ------------------------------------------------------------------

    def _remove_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        self._log(f"Removed duplicates: {removed}")
        return df

    def _handle_missing(self, df):
        """
        Simple missing value handling:
            - Numeric columns → fill with median
            - Categorical columns → fill with mode
        """
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                median = df[col].median()
                df[col] = df[col].fillna(median)
                self._log(f"Filled missing numeric values in '{col}' with median={median}")
            else:
                mode = df[col].mode().iloc[0]
                df[col] = df[col].fillna(mode)
                self._log(f"Filled missing categorical values in '{col}' with mode='{mode}'")

        return df

    def _normalize_types(self, df):
        """
        Normalize data types:
            - Convert numeric-looking strings to numbers
            - Convert boolean-like strings to bool
        """
        for col in df.columns:
            # Try numeric conversion
            try:
                df[col] = pd.to_numeric(df[col])
                self._log(f"Normalized type: '{col}' → numeric")
                continue
            except Exception:
                pass

            # Try boolean conversion
            if df[col].astype(str).str.lower().isin(["true", "false"]).any():
                df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})
                self._log(f"Normalized type: '{col}' → boolean")
                continue

        return df

    def _fix_outliers(self, df):
        """
        Detect and cap outliers using the IQR rule:
            - Values below Q1 - 1.5*IQR → capped to lower bound
            - Values above Q3 + 1.5*IQR → capped to upper bound
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            before = df[col].copy()
            df[col] = df[col].clip(lower, upper)

            changed = (before != df[col]).sum()
            if changed > 0:
                self._log(f"Capped {changed} outliers in '{col}'")

        return df

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(self, message: str):
        if self.logger:
            self.logger.info(f"Cleaning: {message}")
