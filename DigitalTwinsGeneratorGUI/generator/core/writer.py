# generator/core/writer.py

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ChunkWriter:
    """
    Handles chunked writing of telemetry data to CSV or Parquet.

    Features:
        - CSV append mode with header written only once
        - Parquet append mode using PyArrow
        - Automatic file initialization
        - Safe for large-scale generation (GB-level)

    Parameters:
        file_path: str
            Output file path (e.g., "telemetry_output.csv")
        file_format: str
            "csv" or "parquet"
        chunk_size_rows: int
            Number of rows per write cycle
    """

    def __init__(self, file_path: str, file_format: str, chunk_size_rows: int):
        self.file_path = file_path
        self.file_format = file_format.lower()
        self.chunk_size_rows = chunk_size_rows

        # Internal state
        self._csv_header_written = False
        self._parquet_writer = None

        # Remove existing file to avoid mixing old data
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def write_chunk(self, df: pd.DataFrame):
        """
        Writes a DataFrame chunk to disk in the configured format.
        """
        if self.file_format == "csv":
            self._write_csv(df)
        elif self.file_format == "parquet":
            self._write_parquet(df)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

    # ---------------------------------------------------------
    # CSV Writer
    # ---------------------------------------------------------
    def _write_csv(self, df: pd.DataFrame):
        """
        Appends a chunk to a CSV file.
        Header is written only once.
        """
        df.to_csv(
            self.file_path,
            mode="a",
            header=not self._csv_header_written,
            index=False
        )
        self._csv_header_written = True

    # ---------------------------------------------------------
    # Parquet Writer
    # ---------------------------------------------------------
    def _write_parquet(self, df: pd.DataFrame):
        """
        Appends a chunk to a Parquet file using PyArrow.
        Creates a ParquetWriter on first write.
        """
        table = pa.Table.from_pandas(df)

        if self._parquet_writer is None:
            # First write → create writer
            self._parquet_writer = pq.ParquetWriter(
                self.file_path,
                table.schema,
                compression="snappy"
            )

        self._parquet_writer.write_table(table)

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    def close(self):
        """
        Closes Parquet writer if needed.
        Called automatically by TelemetryGenerator.stop().
        """
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None