# generator/core/config_writer.py

import json
from typing import Dict, Any


def write_config(config: Dict[str, Any], schema: Any, output_path: str):
    """
    Writes the shared config.json for the Analyzer.

    Expected structure (new schema):
    {
        "output": {
            "file_path": "...",
            "file_format": "...",
            "estimated_size_gb": ...,
            "chunk_size_rows": ...
        },
        "sampling": {
            "frequency_hz": ...
        },
        "schema": {
            "columns": [...]
        },
        "alerts": {
            "socket_enabled": true,
            "socket_host": "127.0.0.1",
            "socket_port": 5050
        }
    }
    """

    config_dict = {
        "output": {
            "file_path": config["output"]["file_path"],
            "file_format": config["output"]["file_format"],
            "estimated_size_gb": config["output"]["estimated_size_gb"],
            "chunk_size_rows": config["output"]["chunk_size_rows"],
        },
        "sampling": {
            "frequency_hz": config["sampling"]["frequency_hz"],
        },
        "schema": {
            "columns": schema,
        },
        "alerts": config["alerts"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)