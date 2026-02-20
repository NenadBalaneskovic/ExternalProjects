# analyzer/core/config_loader.py

import json
import os
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    """
    Loads the shared config.json file written by the Generator.

    Responsibilities:
        - Validate that the file exists
        - Parse JSON safely
        - Provide clear error messages
        - Return a dictionary with configuration values

    Expected structure of config.json:
        {
            "file_path": "telemetry.parquet",
            "columns": ["Temperature", "Motor RPM", "Error Code"],
            "sampling_rate_hz": 10,
            "timestamp_format": "ISO8601",
            "alerts": {
                "socket_host": "127.0.0.1",
                "socket_port": 5050,
                "socket_enabled": true
            }
        }

    Returns:
        dict with configuration values

    Raises:
        FileNotFoundError
        ValueError (invalid JSON)
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

    # Basic validation
    if "output" not in config or "file_path" not in config["output"]:
        raise ValueError("config.json missing required field: output.file_path")

    # Optional defaults
    config.setdefault("columns", [])
    config.setdefault("sampling_rate_hz", 1)
    config.setdefault("timestamp_format", "ISO8601")
    config.setdefault("alerts", {
        "socket_host": "127.0.0.1",
        "socket_port": 5050,
        "socket_enabled": True
    })

    return config