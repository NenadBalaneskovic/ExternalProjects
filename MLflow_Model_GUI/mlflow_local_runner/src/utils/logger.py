# mlflow_local_runner/src/utils/logger.py
"""
logger.py – Zentrales Logging-System für MLflow Local Runner
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import os

from rich.logging import RichHandler

from utils.paths import get_log_dir

_LOGGING_INITIALIZED = False


def _force_utf8_console():
    """
    Erzwingt UTF-8 für stdout/stderr unter Windows.
    Verhindert UnicodeEncodeError in RichHandler.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Windows Codepage auf UTF-8 setzen
    try:
        if os.name == "nt":
            os.system("chcp 65001 > NUL")
    except Exception:
        pass


def _initialize_logging():
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    _force_utf8_console()

    log_dir = Path(get_log_dir())
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "mlflow_local_runner.log"

    # ---------------------------------------------------------
    # Logging-Konfiguration (Rich + Datei)
    # ---------------------------------------------------------

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            # Rich-Handler (Unicode-sicher)
            RichHandler(
                rich_tracebacks=False,     # verhindert Unicode in Tracebacks
                markup=False,              # verhindert Unicode-Markup
                log_time_format="[%H:%M:%S]"
            ),

            # Log-Datei (UTF-8)
            RotatingFileHandler(
                log_file,
                maxBytes=5_000_000,
                backupCount=3,
                encoding="utf-8"
            )
        ]
    )

    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    _initialize_logging()
    return logging.getLogger(name)
