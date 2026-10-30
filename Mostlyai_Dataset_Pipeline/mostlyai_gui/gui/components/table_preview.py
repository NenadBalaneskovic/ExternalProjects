"""
table_preview.py

A reusable DataFrame preview widget for the Data Privacy Workbench GUI.

This component provides:
    - A title label
    - A scrollable QTableWidget
    - A clean API: update_table(df)
    - Automatic column sizing
    - Graceful handling of empty DataFrames

It is used in:
    - CleaningTab
    - AnonymizationTab
    - PseudonymizationTab
    - SyntheticTab

Author: Nenad
Date: May 2026
"""

import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
)
from PySide6.QtCore import Qt


class TablePreview(QWidget):
    """
    A simple widget that displays a pandas DataFrame in a QTableWidget.

    Parameters
    ----------
    title : str
        Title displayed above the table.
    max_rows : int
        Maximum number of rows to display (default: 50).
    """

    def __init__(self, title="Preview", max_rows=50):
        super().__init__()

        self.title = title
        self.max_rows = max_rows

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Title label
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.title_label)

        # Table widget
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)

        layout.addWidget(self.table)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_table(self, df: pd.DataFrame):
        """
        Update the table to display the given DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to display.
        """
        if df is None or df.empty:
            self._clear_table()
            self.title_label.setText(f"{self.title} (no data)")
            return

        # Limit rows for performance
        df_preview = df.head(self.max_rows)

        # Set table dimensions
        self.table.setRowCount(len(df_preview))
        self.table.setColumnCount(len(df_preview.columns))
        self.table.setHorizontalHeaderLabels(df_preview.columns)

        # Fill table
        for row_idx, (_, row) in enumerate(df_preview.iterrows()):
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setFlags(Qt.ItemIsEnabled)
                self.table.setItem(row_idx, col_idx, item)

        # Auto-size columns
        self.table.resizeColumnsToContents()

        # Update title with row count
        total_rows = len(df)
        self.title_label.setText(f"{self.title} (showing {len(df_preview)} of {total_rows} rows)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clear_table(self):
        """Clear the table contents."""
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
