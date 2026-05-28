"""
cleaning_tab.py

Implements the Data Cleaning tab of the Data Privacy Workbench GUI.
This tab allows the user to:

    - Select a dataset (CSV) or fall back to the default sample dataset
    - Configure cleaning options (duplicates, missing values, types, outliers)
    - Run the cleaning pipeline (delegated to core.cleaning)
    - Preview the cleaned dataset
    - Save the cleaned dataset to a chosen folder

The cleaned dataset is emitted via a Qt signal so that other tabs
(Anonymization, Pseudonymization, Synthetic Data) can consume it.

Author: Nenad
Date: May 2026
"""

import os
import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QCheckBox,
    QGroupBox,
    QLineEdit,
    QMessageBox,
)
from PySide6.QtCore import Signal, Qt

from gui.components.table_preview import TablePreview
from core.cleaning import CleaningPipeline


class CleaningTab(QWidget):
    """
    Data Cleaning tab widget.

    Parameters
    ----------
    config : dict
        Application configuration (paths, defaults).
    logger : logging.Logger
        Shared logger instance.
    status_callback : callable
        Function to update the main window's status bar.
    log_callback : callable
        Function to forward log messages to the Logs tab.
    """

    cleaned_dataset_ready = Signal(object)  # pandas.DataFrame

    def __init__(self, config, logger, status_callback, log_callback):
        super().__init__()

        self.config = config
        self.logger = logger
        self.status_callback = status_callback
        self.log_callback = log_callback

        self.dataset_path = None
        self.cleaned_df = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --------------------------------------------------------------
        # Dataset selection
        # --------------------------------------------------------------
        dataset_box = QGroupBox("Dataset Selection")
        dataset_layout = QHBoxLayout()

        self.dataset_label = QLabel("No dataset selected.")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._select_dataset)

        dataset_layout.addWidget(self.dataset_label)
        dataset_layout.addWidget(browse_btn)
        dataset_box.setLayout(dataset_layout)

        # --------------------------------------------------------------
        # Cleaning options
        # --------------------------------------------------------------
        options_box = QGroupBox("Cleaning Options")
        options_layout = QVBoxLayout()

        self.chk_duplicates = QCheckBox("Remove duplicates")
        self.chk_missing = QCheckBox("Handle missing values")
        self.chk_types = QCheckBox("Normalize data types")
        self.chk_outliers = QCheckBox("Detect & fix outliers")

        # Default: all enabled
        for chk in (self.chk_duplicates, self.chk_missing, self.chk_types, self.chk_outliers):
            chk.setChecked(True)

        options_layout.addWidget(self.chk_duplicates)
        options_layout.addWidget(self.chk_missing)
        options_layout.addWidget(self.chk_types)
        options_layout.addWidget(self.chk_outliers)

        self.clean_btn = QPushButton("Start Cleaning")
        self.clean_btn.clicked.connect(self._run_cleaning)

        options_layout.addWidget(self.clean_btn)
        options_box.setLayout(options_layout)

        # --------------------------------------------------------------
        # Preview table
        # --------------------------------------------------------------
        self.preview = TablePreview(title="Cleaned Data Preview")

        # --------------------------------------------------------------
        # Save cleaned data
        # --------------------------------------------------------------
        save_box = QGroupBox("Save Cleaned Data")
        save_layout = QHBoxLayout()

        self.save_path_edit = QLineEdit(self.config.get("default_cleaned_output", "output/cleaned"))
        save_btn = QPushButton("Save Data")
        save_btn.clicked.connect(self._save_cleaned_data)

        save_layout.addWidget(QLabel("Save to Folder:"))
        save_layout.addWidget(self.save_path_edit)
        save_layout.addWidget(save_btn)
        save_box.setLayout(save_layout)

        # --------------------------------------------------------------
        # Add everything to main layout
        # --------------------------------------------------------------
        layout.addWidget(dataset_box)
        layout.addWidget(options_box)
        layout.addWidget(self.preview)
        layout.addWidget(save_box)

    # ------------------------------------------------------------------
    # Dataset selection
    # ------------------------------------------------------------------

    def _select_dataset(self):
        """Open a file dialog to select a CSV dataset."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "CSV Files (*.csv)"
        )

        if path:
            self.dataset_path = path
            self.dataset_label.setText(os.path.basename(path))
            self.log_callback(f"Dataset selected: {path}")
        else:
            self.log_callback("Dataset selection canceled.")

    # ------------------------------------------------------------------
    # Cleaning pipeline
    # ------------------------------------------------------------------

    def _load_dataset(self) -> pd.DataFrame:
        """
        Load the selected dataset or fall back to the default sample dataset.
        """
        if self.dataset_path and os.path.exists(self.dataset_path):
            self.log_callback(f"Loading dataset: {self.dataset_path}")
            return pd.read_csv(self.dataset_path)

        # Fallback to default dataset
        default_path = "assets/datasets/sample_census.csv"
        self.log_callback("No dataset selected. Using default sample dataset.")
        self.dataset_label.setText("sample_census.csv")
        return pd.read_csv(default_path)

    def _run_cleaning(self):
        """Execute the cleaning pipeline."""
        try:
            df = self._load_dataset()

            pipeline = CleaningPipeline(
                remove_duplicates=self.chk_duplicates.isChecked(),
                handle_missing=self.chk_missing.isChecked(),
                normalize_types=self.chk_types.isChecked(),
                fix_outliers=self.chk_outliers.isChecked(),
                logger=self.logger,
            )

            self.log_callback("Starting cleaning pipeline...")
            cleaned_df = pipeline.run(df)

            self.cleaned_df = cleaned_df
            self.preview.update_table(cleaned_df)

            self.cleaned_dataset_ready.emit(cleaned_df)

            self.status_callback("Cleaning completed successfully.")
            self.log_callback("Cleaning completed successfully.")

        except Exception as e:
            self.logger.error(f"Cleaning failed: {e}")
            QMessageBox.critical(self, "Cleaning Error", f"An error occurred:\n{e}")

    # ------------------------------------------------------------------
    # Save cleaned data
    # ------------------------------------------------------------------

    def _save_cleaned_data(self):
        """Save the cleaned dataset to the selected folder."""
        if self.cleaned_df is None:
            QMessageBox.warning(self, "No Data", "No cleaned dataset available.")
            return

        folder = self.save_path_edit.text().strip()
        os.makedirs(folder, exist_ok=True)

        output_path = os.path.join(folder, "cleaned_dataset.csv")
        self.cleaned_df.to_csv(output_path, index=False)

        self.log_callback(f"Cleaned dataset saved to: {output_path}")
        self.status_callback("Cleaned dataset saved.")
