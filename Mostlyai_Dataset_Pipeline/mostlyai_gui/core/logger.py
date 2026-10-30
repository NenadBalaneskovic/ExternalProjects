"""
logger.py

Central logging system for the Data Privacy Workbench GUI.

This module provides:
    - A single global logging configuration
    - Rich-powered colored console logs
    - Rotating file logs (5 MB, 3 backups)
    - Thread-safe initialization
    - A clean get_logger(name) API
    - Automatic creation of the log directory

The logger is used by:
    - Cleaning pipeline
    - Anonymization pipeline
    - Pseudonymization pipeline
    - Synthetic data engine
    - GUI tabs
    - AppWindow

Author: Nenad
Date: May 2026
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

from rich.logging import RichHandler

# ----------------------------------------------------------------------
# Global initialization guard
# ----------------------------------------------------------------------

_LOGGING_INITIALIZED = False
_LOGGING_LOCK = Lock()


# ----------------------------------------------------------------------
# Helper: determine log directory
# ----------------------------------------------------------------------

def get_log_dir() -> str:
    """
    Return the platform‑appropriate log directory.

    Windows:
        C:/Users/<User>/AppData/Roaming/data_privacy_workbench/logs/

    Linux/macOS:
        ~/.config/data_privacy_workbench/logs/
    """
    if Path.home().joinpath("AppData").exists():  # Windows heuristic
        base = Path.home() / "AppData" / "Roaming" / "data_privacy_workbench"
    else:
        base = Path.home() / ".config" / "data_privacy_workbench"

    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


# ----------------------------------------------------------------------
# Logging initialization
# ----------------------------------------------------------------------

def _initialize_logging():
    """
    Initialize the global logging configuration.
    This function is thread-safe and runs only once.
    """
    global _LOGGING_INITIALIZED

    with _LOGGING_LOCK:
        if _LOGGING_INITIALIZED:
            return

        log_dir = Path(get_log_dir())
        log_file = log_dir / "application.log"

        # --------------------------------------------------------------
        # Configure logging
        # --------------------------------------------------------------
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                # Rich console output
                RichHandler(
                    rich_tracebacks=True,
                    markup=True,
                    log_time_format="[%H:%M:%S]",
                ),

                # Rotating log file
                RotatingFileHandler(
                    log_file,
                    maxBytes=5_000_000,   # 5 MB
                    backupCount=3,
                    encoding="utf-8",
                ),
            ],
        )

        _LOGGING_INITIALIZED = True


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger instance with the global configuration applied.

    Parameters
    ----------
    name : str
        Logger name (usually __name__).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    _initialize_logging()
    return logging.getLogger(name)
