"""
utils/logging_utils.py

Responsible for:
- Providing a unified logging configuration for PyDependencyInspector.
- Offering an in-memory log buffer for the GUI Log Panel.
- Supporting optional file-based logging for debugging and build reports.
- Ensuring deterministic, thread-safe logging behavior.

This module is intentionally:
- GUI-agnostic.
- Pure Python.
- Safe in offline environments.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional


# ---------------------------------------------------------------------------
# In-memory log buffer for GUI
# ---------------------------------------------------------------------------

class InMemoryLogBuffer(logging.Handler):
    """
    A thread-safe in-memory log buffer.

    The GUI can attach to this buffer to display logs in real time.

    Features:
    - Stores log messages in a list.
    - Thread-safe append operations.
    - Optional callback for real-time GUI updates.
    """

    def __init__(self, callback: Optional[callable] = None) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._messages: List[str] = []
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        with self._lock:
            self._messages.append(msg)

        # Notify GUI if callback is provided
        if self._callback:
            try:
                self._callback(msg)
            except Exception:
                pass  # GUI errors should not break logging

    def get_messages(self) -> List[str]:
        """Return a copy of all log messages."""
        with self._lock:
            return list(self._messages)

    def clear(self) -> None:
        """Clear the log buffer."""
        with self._lock:
            self._messages.clear()


# ---------------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------------

def configure_logging(
    level: int = logging.INFO,
    enable_file_logging: bool = False,
    log_file_path: str = "build/logs/pydependencyinspector.log",
    gui_callback: Optional[callable] = None,
) -> InMemoryLogBuffer:
    """
    Configure global logging for PyDependencyInspector.

    :param level: Logging level (default: INFO)
    :param enable_file_logging: Whether to write logs to a file
    :param log_file_path: Path to the log file
    :param gui_callback: Optional callback for GUI log updates
    :return: InMemoryLogBuffer instance
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # In-memory buffer for GUI
    buffer_handler = InMemoryLogBuffer(callback=gui_callback)
    buffer_handler.setFormatter(formatter)
    logger.addHandler(buffer_handler)

    # Optional file logging
    if enable_file_logging:
        try:
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as exc:
            logger.error("Failed to initialize file logging: %s", exc)

    return buffer_handler


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with the global configuration applied.

    Example:
        log = get_logger(__name__)
        log.info("Hello")
    """
    return logging.getLogger(name)
