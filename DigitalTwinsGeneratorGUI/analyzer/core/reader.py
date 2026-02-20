# analyzer/core/reader.py

import os
import pandas as pd
from typing import Optional


class TelemetryReader:
    """
    Efficient tail-reader for CSV or Parquet telemetry files.

    Responsibilities:
        - Track how many rows have already been processed
        - Load only *new* rows on each refresh
        - Support both CSV and Parquet formats
        - Avoid re-reading the entire file (critical for large files)

    Methods:
        read_new_rows() -> pd.DataFrame | None
            Returns only the newly appended rows since last read.
    """

    def __init__(self, file_path: str, row_limit: int = 10_000):
        self.file_path = file_path
        self.row_limit = row_limit

        # Internal state
        self.last_row_index = 0
        self.file_format = self._detect_format(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Telemetry file not found: {file_path}")

    # ---------------------------------------------------------
    # Format detection
    # ---------------------------------------------------------
    def _detect_format(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return "csv"
        if ext == ".parquet":
            return "parquet"
        raise ValueError(f"Unsupported file format: {ext}")

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def read_new_rows(self) -> Optional[pd.DataFrame]:
        """
        Reads only the newly appended rows since the last call.

        Returns:
            pd.DataFrame with new rows
            or None if file is empty or unchanged
        """
        if self.file_format == "csv":
            return self._read_new_csv_rows()
        else:
            return self._read_new_parquet_rows()

    # ---------------------------------------------------------
    # CSV Reader (efficient tail read)
    # ---------------------------------------------------------
    def _read_new_csv_rows(self) -> Optional[pd.DataFrame]:
        """
        Efficiently reads only new rows from a CSV file.
        """
        try:
            # Count total rows in file
            with open(self.file_path, "r", encoding="utf-8") as f:
                total_rows = sum(1 for _ in f)

            # Subtract header
            total_rows -= 1

            if total_rows <= self.last_row_index:
                return None  # No new data

            # Determine how many rows to read
            rows_to_read = min(
                total_rows - self.last_row_index,
                self.row_limit
            )

            skip_rows = range(1, self.last_row_index + 1)
            df = pd.read_csv(
                self.file_path,
                skiprows=skip_rows,
                nrows=rows_to_read,
            )

            # Update internal pointer
            self.last_row_index += len(df)

            return df

        except Exception as e:
            raise RuntimeError(f"CSV read error: {e}")

    # ---------------------------------------------------------
    # Parquet Reader (efficient row slicing)
    # ---------------------------------------------------------
    def _read_new_parquet_rows(self) -> Optional[pd.DataFrame]:
        """
        Efficiently reads only new rows from a Parquet file.
        """
        try:
            # Load metadata only
            meta = pd.read_parquet(self.file_path, columns=[])

            total_rows = meta.shape[0]

            if total_rows <= self.last_row_index:
                return None  # No new data

            rows_to_read = min(
                total_rows - self.last_row_index,
                self.row_limit
            )

            # Read only the required slice
            df = pd.read_parquet(
                self.file_path,
                engine="pyarrow",
                filters=None,
            ).iloc[self.last_row_index:self.last_row_index + rows_to_read]

            self.last_row_index += len(df)

            return df

        except Exception as e:
            raise RuntimeError(f"Parquet read error: {e}")