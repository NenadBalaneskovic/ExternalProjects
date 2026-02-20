# generator/utils/file_tracker.py

import os


class FileTracker:
    """
    Tracks file size and (optionally) row count for the output file.

    Used by:
        - TelemetryGenerator (to update progress)
        - StatusBar (to show file size progress)
        - Analyzer (optional future use)

    Features:
        - Tracks file size in bytes
        - Converts to KB, MB, GB
        - Safe for large files (10–50 GB)
        - Non-blocking, lightweight

    Parameters:
        file_path: str
            Path to the output CSV/Parquet file
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.size_bytes = 0

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def update(self):
        """
        Updates the internal file size counter.
        Called after each chunk write.
        """
        if os.path.exists(self.file_path):
            self.size_bytes = os.path.getsize(self.file_path)
        else:
            self.size_bytes = 0

    def get_size_bytes(self) -> int:
        return self.size_bytes

    def get_size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def get_size_gb(self) -> float:
        return self.size_bytes / (1024 * 1024 * 1024)

    # ---------------------------------------------------------
    # Optional: Row Count Estimation (CSV only)
    # ---------------------------------------------------------
    def estimate_row_count(self) -> int:
        """
        Estimates row count for CSV files by counting newline characters.
        Not used for Parquet.

        This is optional and not called by default.
        """
        if not os.path.exists(self.file_path):
            return 0

        count = 0
        with open(self.file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                count += chunk.count(b"\n")

        # Subtract header row
        return max(0, count - 1)