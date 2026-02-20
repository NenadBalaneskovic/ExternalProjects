# generator/utils/preview_buffer.py

from collections import deque
from typing import Dict, Any, List


class PreviewBuffer:
    """
    Rolling buffer for storing recent preview samples.

    Purpose:
        - Store the last N preview samples for each column
        - Provide fast append and fast retrieval
        - Avoid unbounded memory growth
        - Decouple preview storage from the GUI and generator

    Parameters:
        max_points: int
            Maximum number of samples to keep per column
    """

    def __init__(self, max_points: int = 500):
        self.max_points = max_points
        self.buffers: Dict[str, deque] = {}

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def add_sample(self, sample: Dict[str, Any]):
        """
        Adds a new sample to the buffer.

        Args:
            sample: dict
                Example:
                    {
                        "Temperature": 42.1,
                        "Motor RPM": 1500,
                        "Voltage": 230.5
                    }
        """
        for col, value in sample.items():
            if col not in self.buffers:
                self.buffers[col] = deque(maxlen=self.max_points)
            self.buffers[col].append(value)

    def get_series(self, column: str) -> List[Any]:
        """
        Returns the rolling series for a given column.

        Args:
            column: str
                Column name

        Returns:
            list of recent values (up to max_points)
        """
        if column not in self.buffers:
            return []
        return list(self.buffers[column])

    def clear(self):
        """
        Clears all buffers.
        """
        self.buffers.clear()