"""
synthetic.py (SDV version)

Offline synthetic data generator using SDV.

This module replaces the MOSTLY AI Client Mode engine with a fully local,
open-source generator based on SDV's CTGAN model.

It supports:

    - Training a CTGAN model on a dataset
    - Generating synthetic rows
    - Returning results as pandas DataFrames
    - Logging progress to the GUI

The GUI interacts with this engine through SyntheticTab.

Author: Nenad
Date: May 2026
"""

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


class SDVSyntheticEngine:
    """
    Offline synthetic data generator using SDV (CTGAN).

    Parameters
    ----------
    logger : logging.Logger
        Logger instance for reporting progress.
    """

    def __init__(self, logger=None):
        self.logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_generator(self, df: pd.DataFrame, name: str = "sdv_model") -> CTGANSynthesizer:
        """
        Train a CTGAN model on the provided dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            Training dataset.
        name : str
            Model name (for logging only).

        Returns
        -------
        CTGANSynthesizer
            Trained SDV model.
        """
        self._log(f"Preparing metadata for dataset '{name}'...")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        self._log(f"Training SDV CTGAN model '{name}'...")

        model = CTGANSynthesizer(metadata)
        model.fit(df)

        self._log(f"Model '{name}' trained successfully.")
        return model

    def generate_synthetic_data(self, model: CTGANSynthesizer, num_rows: int) -> pd.DataFrame:
        """
        Generate synthetic data using a trained SDV model.

        Parameters
        ----------
        model : CTGANSynthesizer
            Trained SDV model.
        num_rows : int
            Number of synthetic rows to generate.

        Returns
        -------
        pandas.DataFrame
            Synthetic dataset.
        """
        self._log(f"Generating {num_rows} synthetic rows...")

        synthetic_df = model.sample(num_rows)

        self._log("Synthetic data generation completed.")
        return synthetic_df

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(self, message: str):
        if self.logger:
            self.logger.info(f"Synthetic (SDV): {message}")
