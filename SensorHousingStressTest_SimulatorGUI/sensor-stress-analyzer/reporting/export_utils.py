# reporting/export_utils.py
"""
Export utilities for Sensor Stress Analyzer.
Handles saving results in JSON, TXT, and CSV formats.
"""

import json
import csv
import numpy as np
from config import RESULTS_FILENAME, TEXT_REPORT_FILENAME, CSV_EXPORT_FILENAME


def _sanitize_for_json(obj):
    """Convert NumPy arrays and other non-serializable objects to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    # Convert NumPy scalar types to native Python scalars
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def save_results(results: dict, filename: str = RESULTS_FILENAME):
    """Save results dictionary to a JSON file, converting arrays to lists."""
    safe_results = _sanitize_for_json(results)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(safe_results, f, indent=4)
    return filename


def save_text_report(results: dict, filename: str = TEXT_REPORT_FILENAME):
    """Save results dictionary to a plain text summary file."""
    safe_results = _sanitize_for_json(results)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Sensor Stress Analyzer Results\n")
        f.write("=" * 40 + "\n\n")
        for key, value in safe_results.items():
            f.write(f"{key}: {value}\n")
    return filename


def _write_array_section(writer: csv.writer, title: str, data):
    """
    Write array-like or dict-like data to CSV safely.
    Handles:
    - 1D lists of scalars
    - 2D lists (list of lists/tuples)
    - dicts of scalars or lists
    """
    writer.writerow([])
    writer.writerow([title])

    # Dict: write key-value, expanding lists when present
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                # If list of scalars, write as row with key prefix
                if all(not isinstance(x, (list, tuple)) for x in v):
                    writer.writerow([k] + list(v))
                else:
                    # Nested list: write each row with key prefix
                    for row in v:
                        if isinstance(row, (list, tuple)):
                            writer.writerow([k] + list(row))
                        else:
                            writer.writerow([k, row])
            else:
                writer.writerow([k, v])
        return

    # List/tuple: detect 1D vs 2D
    if isinstance(data, (list, tuple)):
        # If list of scalars (1D), write each scalar as its own row
        if all(not isinstance(x, (list, tuple)) for x in data):
            for x in data:
                writer.writerow([x])
            return
        # Otherwise, treat as 2D-like
        for row in data:
            if isinstance(row, (list, tuple)):
                writer.writerow(list(row))
            else:
                writer.writerow([row])
        return

    # Fallback: single scalar value
    writer.writerow([data])


def save_csv(results: dict, filename: str = CSV_EXPORT_FILENAME):
    """Save results dictionary to a CSV file, including arrays as tables."""
    safe_results = _sanitize_for_json(results)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 1) Write scalar values
        for key, value in safe_results.items():
            if isinstance(value, (int, float, str)):
                writer.writerow([key, value])

        # 2) Write known array/dataset sections safely
        if "vertices" in safe_results:
            _write_array_section(writer, "Vertices (x,y)", safe_results["vertices"])

        if "forces" in safe_results:
            _write_array_section(writer, "Forces (fx,fy)", safe_results["forces"])

        if "stress_map" in safe_results:
            _write_array_section(writer, "Stress Map", safe_results["stress_map"])

        if "heat_map" in safe_results:
            _write_array_section(writer, "Heat Map", safe_results["heat_map"])

        # 3) Generic catch-all for any remaining list/dict values
        for key, value in safe_results.items():
            if isinstance(value, (list, tuple, dict)) and key not in {
                "vertices", "forces", "stress_map", "heat_map"
            }:
                _write_array_section(writer, key, value)

    return filename
