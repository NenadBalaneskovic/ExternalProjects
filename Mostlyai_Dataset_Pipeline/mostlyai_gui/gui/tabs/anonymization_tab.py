"""
anonymization_tab.py

Implements the Anonymization tab of the Data Privacy Workbench GUI.
This tab allows the user to:

    - Use the cleaned dataset from the Cleaning tab OR load a CSV manually
    - Configure anonymization rules per column
    - Run the anonymization pipeline (delegated to core.anonymization)
    - Preview the anonymized dataset
    - Save the anonymized dataset to a chosen folder

The anonymized dataset is emitted via a Qt signal so that other tabs
(e.g., Pseudonymization, Synthetic Data) can consume it.

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
    QGroupBox,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
)
from PySide6.QtCore import Signal, Qt

from gui.components.table_preview import TablePreview
from core.anonymization import AnonymizationPipeline


class AnonymizationTab(QWidget):
    """
    Anonymization tab widget.

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

    anonymized_dataset_ready = Signal(object)  # pandas.DataFrame

    def __init__(self, config, logger, status_callback, log_callback):
        super().__init__()

        self.config = config
        self.logger = logger
        self.status_callback = status_callback
        self.log_callback = log_callback

        self.cleaned_df = None
        self.anonymized_df = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --------------------------------------------------------------
        # Input source selection
        # --------------------------------------------------------------
        input_box = QGroupBox("Input Dataset")
        input_layout = QHBoxLayout()

        self.input_label = QLabel("No dataset loaded.")
        load_btn = QPushButton("Load CSV...")
        load_btn.clicked.connect(self._load_external_dataset)

        input_layout.addWidget(self.input_label)
        input_layout.addWidget(load_btn)
        input_box.setLayout(input_layout)

        # --------------------------------------------------------------
        # Anonymization rules table
        # --------------------------------------------------------------
        rules_box = QGroupBox("Anonymization Rules")
        rules_layout = QVBoxLayout()

        self.rules_table = QTableWidget()
        self.rules_table.setColumnCount(2)
        self.rules_table.setHorizontalHeaderLabels(["Field", "Action"])
        self.rules_table.horizontalHeader().setStretchLastSection(True)

        rules_layout.addWidget(self.rules_table)
        rules_box.setLayout(rules_layout)

        # --------------------------------------------------------------
        # Run anonymization
        # --------------------------------------------------------------
        run_btn = QPushButton("Run Anonymization")
        run_btn.clicked.connect(self._run_anonymization)

        # --------------------------------------------------------------
        # Preview table
        # --------------------------------------------------------------
        self.preview = TablePreview(title="Anonymized Data Preview")

        # --------------------------------------------------------------
        # Save anonymized data
        # --------------------------------------------------------------
        save_box = QGroupBox("Save Anonymized Data")
        save_layout = QHBoxLayout()

        self.save_path_edit = QLineEdit(self.config.get("default_anonymized_output", "output/anonymized"))
        save_btn = QPushButton("Save Data")
        save_btn.clicked.connect(self._save_anonymized_data)

        save_layout.addWidget(QLabel("Save to Folder:"))
        save_layout.addWidget(self.save_path_edit)
        save_layout.addWidget(save_btn)
        save_box.setLayout(save_layout)

        # --------------------------------------------------------------
        # Add everything to main layout
        # --------------------------------------------------------------
        layout.addWidget(input_box)
        layout.addWidget(rules_box)
        layout.addWidget(run_btn)
        layout.addWidget(self.preview)
        layout.addWidget(save_box)

    # ------------------------------------------------------------------
    # Receiving cleaned dataset from CleaningTab
    # ------------------------------------------------------------------

    def receive_cleaned_dataset(self, df: pd.DataFrame):
        """Receive cleaned dataset from CleaningTab."""
        self.cleaned_df = df
        self.input_label.setText("Using cleaned dataset")
        self._populate_rules_table(df.columns)
        self.log_callback("AnonymizationTab: Cleaned dataset received.")

    # ------------------------------------------------------------------
    # Load external dataset
    # ------------------------------------------------------------------

    def _load_external_dataset(self):
        """Load a CSV file manually."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "CSV Files (*.csv)"
        )

        if not path:
            self.log_callback("Anonymization: Dataset selection canceled.")
            return

        try:
            df = pd.read_csv(path)
            self.cleaned_df = df
            self.input_label.setText(os.path.basename(path))
            self._populate_rules_table(df.columns)
            self.log_callback(f"Anonymization: Loaded dataset {path}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load dataset:\n{e}")
            self.logger.error(f"Failed to load dataset: {e}")

    # ------------------------------------------------------------------
    # Populate rules table
    # ------------------------------------------------------------------

    def _populate_rules_table(self, columns):
        """Fill the rules table with dataset columns and action dropdowns."""
        self.rules_table.setRowCount(len(columns))

        for row, col_name in enumerate(columns):
            # Field name
            item = QTableWidgetItem(col_name)
            item.setFlags(Qt.ItemIsEnabled)
            self.rules_table.setItem(row, 0, item)

            # Action dropdown
            combo = QComboBox()
            combo.addItems(["mask", "generalize", "remove", "hash", "none"])
            self.rules_table.setCellWidget(row, 1, combo)

    # ------------------------------------------------------------------
    # Run anonymization
    # ------------------------------------------------------------------

    def _run_anonymization(self):
        """Execute the anonymization pipeline."""
        if self.cleaned_df is None:
            QMessageBox.warning(self, "No Data", "No dataset available for anonymization.")
            return

        try:
            rules = self._collect_rules()
            self.log_callback("Starting anonymization pipeline...")

            pipeline = AnonymizationPipeline(
                rules=rules,
                logger=self.logger,
            )

            anonymized_df = pipeline.run(self.cleaned_df)

            self.anonymized_df = anonymized_df
            self.preview.update_table(anonymized_df)

            self.anonymized_dataset_ready.emit(anonymized_df)

            self.status_callback("Anonymization completed successfully.")
            self.log_callback("Anonymization completed successfully.")

        except Exception as e:
            self.logger.error(f"Anonymization failed: {e}")
            QMessageBox.critical(self, "Anonymization Error", f"An error occurred:\n{e}")

    def _collect_rules(self):
        """Extract anonymization rules from the rules table."""
        rules = {}

        for row in range(self.rules_table.rowCount()):
            field = self.rules_table.item(row, 0).text()
            action = self.rules_table.cellWidget(row, 1).currentText()
            rules[field] = action

        return rules

    # ------------------------------------------------------------------
    # Save anonymized data
    # ------------------------------------------------------------------

    def _save_anonymized_data(self):
        """Save the anonymized dataset to the selected folder."""
        if self.anonymized_df is None:
            QMessageBox.warning(self, "No Data", "No anonymized dataset available.")
            return

        folder = self.save_path_edit.text().strip()
        os.makedirs(folder, exist_ok=True)

        output_path = os.path.join(folder, "anonymized_dataset.csv")
        self.anonymized_df.to_csv(output_path, index=False)

        self.log_callback(f"Anonymized dataset saved to: {output_path}")
        self.status_callback("Anonymized dataset saved.")
