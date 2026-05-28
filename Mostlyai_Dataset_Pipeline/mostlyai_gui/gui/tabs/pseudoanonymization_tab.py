"""
pseudonymization_tab.py

Implements the Pseudonymization tab of the Data Privacy Workbench GUI.
This tab allows the user to:

    - Use the cleaned dataset from the Cleaning tab OR load a CSV manually
    - Configure pseudonymization rules per column
    - Optionally store a mapping table (for reversible pseudonymization)
    - Run the pseudonymization pipeline (delegated to core.pseudonymization)
    - Preview the pseudonymized dataset
    - Save the pseudonymized dataset to a chosen folder

The pseudonymized dataset is emitted via a Qt signal so that other tabs
(e.g., Synthetic Data) can consume it.

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
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
)
from PySide6.QtCore import Signal, Qt

from gui.components.table_preview import TablePreview
from core.pseudoanonymization import PseudonymizationPipeline


class PseudonymizationTab(QWidget):
    """
    Pseudonymization tab widget.

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

    pseudonymized_dataset_ready = Signal(object)  # pandas.DataFrame

    def __init__(self, config, logger, status_callback, log_callback):
        super().__init__()

        self.config = config
        self.logger = logger
        self.status_callback = status_callback
        self.log_callback = log_callback

        self.cleaned_df = None
        self.pseudonymized_df = None

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
        # Pseudonymization rules table
        # --------------------------------------------------------------
        rules_box = QGroupBox("Pseudonymization Rules")
        rules_layout = QVBoxLayout()

        self.rules_table = QTableWidget()
        self.rules_table.setColumnCount(2)
        self.rules_table.setHorizontalHeaderLabels(["Field", "Action"])
        self.rules_table.horizontalHeader().setStretchLastSection(True)

        rules_layout.addWidget(self.rules_table)
        rules_box.setLayout(rules_layout)

        # --------------------------------------------------------------
        # Mapping table option
        # --------------------------------------------------------------
        mapping_box = QGroupBox("Mapping Table")
        mapping_layout = QHBoxLayout()

        self.chk_store_mapping = QCheckBox("Store mapping table")
        self.mapping_path_edit = QLineEdit(self.config.get("default_mapping_output", "output/mappings"))
        self.mapping_path_edit.setEnabled(False)

        self.chk_store_mapping.stateChanged.connect(
            lambda state: self.mapping_path_edit.setEnabled(state == Qt.Checked)
        )

        mapping_layout.addWidget(self.chk_store_mapping)
        mapping_layout.addWidget(self.mapping_path_edit)
        mapping_box.setLayout(mapping_layout)

        # --------------------------------------------------------------
        # Run pseudonymization
        # --------------------------------------------------------------
        run_btn = QPushButton("Run Pseudonymization")
        run_btn.clicked.connect(self._run_pseudonymization)

        # --------------------------------------------------------------
        # Preview table
        # --------------------------------------------------------------
        self.preview = TablePreview(title="Pseudonymized Data Preview")

        # --------------------------------------------------------------
        # Save pseudonymized data
        # --------------------------------------------------------------
        save_box = QGroupBox("Save Pseudonymized Data")
        save_layout = QHBoxLayout()

        self.save_path_edit = QLineEdit(self.config.get("default_pseudonymized_output", "output/pseudonymized"))
        save_btn = QPushButton("Save Data")
        save_btn.clicked.connect(self._save_pseudonymized_data)

        save_layout.addWidget(QLabel("Save to Folder:"))
        save_layout.addWidget(self.save_path_edit)
        save_layout.addWidget(save_btn)
        save_box.setLayout(save_layout)

        # --------------------------------------------------------------
        # Add everything to main layout
        # --------------------------------------------------------------
        layout.addWidget(input_box)
        layout.addWidget(rules_box)
        layout.addWidget(mapping_box)
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
        self.log_callback("PseudonymizationTab: Cleaned dataset received.")

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
            self.log_callback("Pseudonymization: Dataset selection canceled.")
            return

        try:
            df = pd.read_csv(path)
            self.cleaned_df = df
            self.input_label.setText(os.path.basename(path))
            self._populate_rules_table(df.columns)
            self.log_callback(f"Pseudonymization: Loaded dataset {path}")
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
            combo.addItems(["pseudonymize", "hash", "none"])
            self.rules_table.setCellWidget(row, 1, combo)

    # ------------------------------------------------------------------
    # Run pseudonymization
    # ------------------------------------------------------------------

    def _run_pseudonymization(self):
        """Execute the pseudonymization pipeline."""
        if self.cleaned_df is None:
            QMessageBox.warning(self, "No Data", "No dataset available for pseudonymization.")
            return

        try:
            rules = self._collect_rules()
            store_mapping = self.chk_store_mapping.isChecked()
            mapping_path = self.mapping_path_edit.text().strip()

            self.log_callback("Starting pseudonymization pipeline...")

            pipeline = PseudonymizationPipeline(
                rules=rules,
                store_mapping=store_mapping,
                mapping_path=mapping_path,
                logger=self.logger,
            )

            pseudonymized_df = pipeline.run(self.cleaned_df)

            self.pseudonymized_df = pseudonymized_df
            self.preview.update_table(pseudonymized_df)

            self.pseudonymized_dataset_ready.emit(pseudonymized_df)

            self.status_callback("Pseudonymization completed successfully.")
            self.log_callback("Pseudonymization completed successfully.")

        except Exception as e:
            self.logger.error(f"Pseudonymization failed: {e}")
            QMessageBox.critical(self, "Pseudonymization Error", f"An error occurred:\n{e}")

    def _collect_rules(self):
        """Extract pseudonymization rules from the rules table."""
        rules = {}

        for row in range(self.rules_table.rowCount()):
            field = self.rules_table.item(row, 0).text()
            action = self.rules_table.cellWidget(row, 1).currentText()
            rules[field] = action

        return rules

    # ------------------------------------------------------------------
    # Save pseudonymized data
    # ------------------------------------------------------------------

    def _save_pseudonymized_data(self):
        """Save the pseudonymized dataset to the selected folder."""
        if self.pseudonymized_df is None:
            QMessageBox.warning(self, "No Data", "No pseudonymized dataset available.")
            return

        folder = self.save_path_edit.text().strip()
        os.makedirs(folder, exist_ok=True)

        output_path = os.path.join(folder, "pseudonymized_dataset.csv")
        self.pseudonymized_df.to_csv(output_path, index=False)

        self.log_callback(f"Pseudonymized dataset saved to: {output_path}")
        self.status_callback("Pseudonymized dataset saved.")
