# analyzer/utils/file_monitor.py

import os
from typing import Optional
import pandas as pd


class FileMonitor:
    """
    Lightweight file monitor for telemetry files.

    Responsibilities:
        - Track file size changes
        - Track row count (CSV or Parquet)
        - Detect file resets or truncation
        - Provide metadata for progress reporting

    Methods:
        update() -> dict
            Returns a dictionary describing the current file state:
                {
                    "size_bytes": int,
                    "rows": int,
                    "reset_detected": bool
                }
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

        # Internal state
        self.last_size: int = 0
        self.last_rows: int = 0

        # Detect format
        self.format = self._detect_format(file_path)

    # ---------------------------------------------------------
    # Format detection
    # ---------------------------------------------------------
    def _detect_format(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return "csv"
        if ext == ".parquet":
            return "parquet"
        raise ValueError(f"Unsupported telemetry file format: {ext}")

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def update(self) -> dict:
        """
        Checks the current file state and returns metadata.

        Returns:
            {
                "size_bytes": int,
                "rows": int,
                "reset_detected": bool
            }
        """
        if not os.path.exists(self.file_path):
            return {
                "size_bytes": 0,
                "rows": 0,
                "reset_detected": False,
            }

        size = os.path.getsize(self.file_path)
        rows = self._count_rows()

        # Detect file reset (e.g., generator restarted)
        reset = size < self.last_size or rows < self.last_rows

        # Update internal state
        self.last_size = size
        self.last_rows = rows

        return {
            "size_bytes": size,
            "rows": rows,
            "reset_detected": reset,
        }

    # ---------------------------------------------------------
    # Row counting
    # ---------------------------------------------------------
    def _count_rows(self) -> int:
        """
        Efficient row counting for CSV and Parquet.
        """
        try:
            if self.format == "csv":
                return self._count_csv_rows()
            else:
                return self._count_parquet_rows()
        except Exception:
            return 0

    def _count_csv_rows(self) -> int:
        """
        Counts rows in a CSV file without loading the full file.
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            # subtract header
            return max(0, sum(1 for _ in f) - 1)

    def _count_parquet_rows(self) -> int:
        """
        Counts rows in a Parquet file using metadata only.
        """
        meta = pd.read_parquet(self.file_path, columns=[])
        return meta.shape[0]