# generator/core/generator.py

import threading
import time
from typing import Callable, List, Dict, Any

import numpy as np
import pandas as pd

from .column_models import COLUMN_MODEL_MAP
from .writer import ChunkWriter
from .config_writer import write_config
from ..utils.file_tracker import FileTracker


class TelemetryGenerator:
    """
    Backend engine for telemetry data generation.

    Responsibilities:
        - Run a generation loop in a background thread
        - Use schema + config to simulate rows in chunks
        - Write chunks to disk (CSV/Parquet)
        - Send small samples to preview callback
        - Update progress via progress callback
        - Emit alerts via alert callback
        - Write shared config.json for the analyzer
    """

    def __init__(
        self,
        schema: List[Dict[str, Any]],
        config: Dict[str, Any],
        preview_callback: Callable[[Dict[str, Any]], None],
        progress_callback: Callable[[float, str], None],
        alert_callback: Callable[[str], None],
    ):
        print("GENERATOR INITIALIZED")

        self.schema = schema
        self.config = config
        self.preview_callback = preview_callback
        self.progress_callback = progress_callback
        self.alert_callback = alert_callback

        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

        # ---------------------------------------------------------
        # Normalize config keys (SettingsPanel → Generator backend)
        # ---------------------------------------------------------
        self.total_rows = config["rows"]
        self.file_format = config["file_format"]
        self.frequency_hz = config["frequency_hz"]
        self.target_gb = config["target_gb"]

        # Chunk size: fixed or dynamic
        self.chunk_size = 10_000

        # Output file path
        self.output_path = (
            "telemetry_output.csv"
            if self.file_format == "csv"
            else "telemetry_output.parquet"
        )

        # ---------------------------------------------------------
        # Writer for CSV/Parquet
        # ---------------------------------------------------------
        self.writer = ChunkWriter(
            file_path=self.output_path,
            file_format=self.file_format,
            chunk_size_rows=self.chunk_size,
        )

        # File tracker for progress estimation
        self.file_tracker = FileTracker(self.output_path)

        # ---------------------------------------------------------
        # Write shared config.json for Analyzer
        # ---------------------------------------------------------
        analyzer_config = {
            "output": {
                "file_path": self.output_path,
                "file_format": self.file_format,
                "estimated_size_gb": self.target_gb,
                "chunk_size_rows": self.chunk_size,
            },
            "sampling": {
                "frequency_hz": self.frequency_hz,
            },
            "schema": {
                "columns": self.schema,
            },
            "alerts": {
                "socket_enabled": True,
                "socket_host": "127.0.0.1",
                "socket_port": 5050,
            },
        }

        write_config(
            config=analyzer_config,
            schema=self.schema,
            output_path="config.json",
        )

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def start(self):
        """Starts the generation loop in a background thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._run_loop_safe,
            daemon=True,
        )
        self._thread.start()
        self.alert_callback("Generator started")

    def _run_loop_safe(self):
        """Wrapper to catch and report exceptions from the generator thread."""
        import os
        print("GENERATOR THREAD PID:", os.getpid())
        try:
            print("ENTERING RUN LOOP")
            self._run_loop()
        except Exception as e:
            print("GENERATOR THREAD CRASHED:", e)
            import traceback
            traceback.print_exc()
            self.alert_callback(f"Generator crashed: {e}")

    def stop(self):
        """Signals the generation loop to stop and waits for thread to finish."""
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()
        self.alert_callback("Generator stopped")

    # ---------------------------------------------------------
    # Internal Loop
    # ---------------------------------------------------------
    def _run_loop(self):
        """Main generation loop."""
        rows_generated = 0
        sleep_interval = 1.0 / self.frequency_hz if self.frequency_hz > 0 else 0.0

        # Throttle preview updates (20 FPS max)
        last_preview_time = time.time()

        while not self._stop_flag.is_set() and rows_generated < self.total_rows:
            print("RUN LOOP STARTED")
            rows_to_generate = min(self.chunk_size, self.total_rows - rows_generated)

            df_chunk = self._generate_chunk(rows_to_generate)

            # Write to disk
            self.writer.write_chunk(df_chunk)

            rows_generated += rows_to_generate

            # Update progress
            percent = 100.0 * rows_generated / self.total_rows
            self.progress_callback(
                percent,
                f"Generated {rows_generated} / {self.total_rows} rows",
            )

            # ---------------------------------------------------------
            # Throttled preview update (max ~20 FPS)
            # ---------------------------------------------------------
            now = time.time()
            if now - last_preview_time >= 0.05:  # 50 ms
                last_row = df_chunk.iloc[-1].to_dict()
                print("CALLING PREVIEW CALLBACK WITH:", last_row)
                self.preview_callback(last_row)
                last_preview_time = now

            # Track file size
            self.file_tracker.update()

            # Real-time pacing
            if sleep_interval > 0:
                time.sleep(sleep_interval)

        self.alert_callback("Generation complete")

    # ---------------------------------------------------------
    # Chunk Generation
    # ---------------------------------------------------------
    def _generate_chunk(self, n_rows: int) -> pd.DataFrame:
        """Generates a DataFrame with n_rows according to the schema."""
        data = {}

        for col in self.schema:
            name = col["name"]
            gen_name = col.get("generator")
            model_func = COLUMN_MODEL_MAP.get(gen_name)

            if model_func is None:
                data[name] = np.zeros(n_rows)
                continue

            data[name] = model_func(n_rows, col, self.config)

        return pd.DataFrame(data)