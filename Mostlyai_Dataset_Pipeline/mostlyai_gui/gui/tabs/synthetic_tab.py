"""
synthetic_tab.py

Implements the Synthetic Data Generation tab of the Data Privacy Workbench GUI.
This tab allows the user to:

    - Use the cleaned dataset from the Cleaning tab OR load a CSV manually
    - Train a new SDV CTGAN model
    - Generate synthetic data locally (offline)
    - Preview the synthetic dataset
    - Save the synthetic dataset to a chosen folder

The synthetic dataset is emitted via a Qt signal so that other tabs
or external modules can consume it.

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
    QLineEdit,
    QComboBox,
    QMessageBox,
    QSpinBox,
)
from PySide6.QtCore import Signal

from gui.components.table_preview import TablePreview
from core.synthetic import SDVSyntheticEngine


class SyntheticTab(QWidget):
    """
    Synthetic Data Generation tab widget.

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

    synthetic_dataset_ready = Signal(object)  # pandas.DataFrame

    def __init__(self, config, logger, status_callback, log_callback):
        super().__init__()

        self.config = config
        self.logger = logger
        self.status_callback = status_callback
        self.log_callback = log_callback

        self.cleaned_df = None
        self.synthetic_df = None
        self.trained_model = None  # SDV model stored in memory

        # SDV engine (no API key needed)
        self.engine = SDVSyntheticEngine(logger=logger)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --------------------------------------------------------------
        # Input dataset selection
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
        # Model training
        # --------------------------------------------------------------
        generator_box = QGroupBox("Model Training (SDV CTGAN)")
        generator_layout = QVBoxLayout()

        self.new_generator_name = QLineEdit()
        self.new_generator_name.setPlaceholderText("Model name")

        self.max_epochs = QSpinBox()
        self.max_epochs.setRange(1, 500)
        self.max_epochs.setValue(20)

        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self._train_generator)

        generator_layout.addWidget(QLabel("Model Name:"))
        generator_layout.addWidget(self.new_generator_name)
        generator_layout.addWidget(QLabel("Training Epochs:"))
        generator_layout.addWidget(self.max_epochs)
        generator_layout.addWidget(self.train_btn)

        generator_box.setLayout(generator_layout)

        # --------------------------------------------------------------
        # Synthetic generation
        # --------------------------------------------------------------
        synth_box = QGroupBox("Generate Synthetic Data")
        synth_layout = QHBoxLayout()

        self.num_rows = QSpinBox()
        self.num_rows.setRange(10, 1_000_000)
        self.num_rows.setValue(1000)

        generate_btn = QPushButton("Generate Synthetic Data")
        generate_btn.clicked.connect(self._generate_synthetic_data)

        synth_layout.addWidget(QLabel("Rows:"))
        synth_layout.addWidget(self.num_rows)
        synth_layout.addWidget(generate_btn)
        synth_box.setLayout(synth_layout)

        # --------------------------------------------------------------
        # Preview table
        # --------------------------------------------------------------
        self.preview = TablePreview(title="Synthetic Data Preview")

        # --------------------------------------------------------------
        # Save synthetic data
        # --------------------------------------------------------------
        save_box = QGroupBox("Save Synthetic Data")
        save_layout = QHBoxLayout()

        self.save_path_edit = QLineEdit(self.config.get("default_synthetic_output", "output/synthetic"))
        save_btn = QPushButton("Save Data")
        save_btn.clicked.connect(self._save_synthetic_data)

        save_layout.addWidget(QLabel("Save to Folder:"))
        save_layout.addWidget(self.save_path_edit)
        save_layout.addWidget(save_btn)
        save_box.setLayout(save_layout)

        # --------------------------------------------------------------
        # Add everything to main layout
        # --------------------------------------------------------------
        layout.addWidget(input_box)
        layout.addWidget(generator_box)
        layout.addWidget(synth_box)
        layout.addWidget(self.preview)
        layout.addWidget(save_box)

    # ------------------------------------------------------------------
    # Receiving cleaned dataset from CleaningTab
    # ------------------------------------------------------------------

    def receive_cleaned_dataset(self, df: pd.DataFrame):
        """Receive cleaned dataset from CleaningTab."""
        self.cleaned_df = df
        self.input_label.setText("Using cleaned dataset")
        self.log_callback("SyntheticTab: Cleaned dataset received.")

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
            self.log_callback("Synthetic: Dataset selection canceled.")
            return

        try:
            df = pd.read_csv(path)
            self.cleaned_df = df
            self.input_label.setText(os.path.basename(path))
            self.log_callback(f"Synthetic: Loaded dataset {path}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load dataset:\n{e}")
            self.logger.error(f"Failed to load dataset: {e}")

    # ------------------------------------------------------------------
    # Train SDV model
    # ------------------------------------------------------------------

    def _train_generator(self):
        """Train a new SDV CTGAN model."""
        if self.cleaned_df is None:
            QMessageBox.warning(self, "No Data", "No dataset available for training.")
            return

        name = self.new_generator_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please enter a model name.")
            return

        try:
            self.log_callback(f"Training SDV model '{name}'...")
            self.trained_model = self.engine.train_generator(
                df=self.cleaned_df,
                name=name,
            )

            self.log_callback(f"Model trained successfully: {name}")
            self.status_callback("Model training completed.")

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            QMessageBox.critical(self, "Training Error", f"An error occurred:\n{e}")

    # ------------------------------------------------------------------
    # Generate synthetic data
    # ------------------------------------------------------------------

    def _generate_synthetic_data(self):
        """Generate synthetic data using the trained SDV model."""
        if self.cleaned_df is None:
            QMessageBox.warning(self, "No Data", "No dataset available for synthetic generation.")
            return

        if self.trained_model is None:
            QMessageBox.warning(self, "No Model", "Train a model first.")
            return

        try:
            rows = self.num_rows.value()
            self.log_callback(f"Generating {rows} synthetic rows...")

            synthetic_df = self.engine.generate_synthetic_data(
                model=self.trained_model,
                num_rows=rows,
            )

            self.synthetic_df = synthetic_df
            self.preview.update_table(synthetic_df)

            self.synthetic_dataset_ready.emit(synthetic_df)

            self.status_callback("Synthetic data generation completed.")
            self.log_callback("Synthetic data generation completed.")

        except Exception as e:
            self.logger.error(f"Synthetic generation failed: {e}")
            QMessageBox.critical(self, "Generation Error", f"An error occurred:\n{e}")

    # ------------------------------------------------------------------
    # Save synthetic data
    # ------------------------------------------------------------------

    def _save_synthetic_data(self):
        """Save the synthetic dataset to the selected folder."""
        if self.synthetic_df is None:
            QMessageBox.warning(self, "No Data", "No synthetic dataset available.")
            return

        folder = self.save_path_edit.text().strip()
        os.makedirs(folder, exist_ok=True)

        output_path = os.path.join(folder, "synthetic_dataset.csv")
        self.synthetic_df.to_csv(output_path, index=False)

        self.log_callback(f"Synthetic dataset saved to: {output_path}")
        self.status_callback("Synthetic dataset saved.")
