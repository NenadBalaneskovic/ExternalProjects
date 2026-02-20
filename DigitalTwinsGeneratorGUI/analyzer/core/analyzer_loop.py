# analyzer/core/analyzer_loop.py

import threading
import time
from typing import Callable, Dict, List, Any, Optional

from .reader import TelemetryReader
from ..modules import statistics, clustering, forecasting, nlp, deep_learning, xai


class AnalyzerLoop:
    """
    Main analysis loop for the Telemetry Analyzer.

    Responsibilities:
        - Periodically read new data from the telemetry file
        - Dispatch selected analysis modules
        - Aggregate results and send them to the GUI
        - Report progress, logs, and health status

    Lifecycle:
        - start(analysis_config, selected_modules)
        - loop in background thread until stop() is called
    """

    def __init__(
        self,
        config: dict,
        progress_callback: Callable[[float, str], None],
        log_callback: Callable[[str], None],
        visualization_callback: Callable[[Dict[str, Any]], None],
        health_callback: Callable[[Dict[str, Any]], None],
    ):
        self.config = config or {}
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.visualization_callback = visualization_callback
        self.health_callback = health_callback

        self.reader: Optional[TelemetryReader] = None
        self.selected_modules: List[str] = []

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Internal state
        self.total_rows_processed = 0

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def start(self, analysis_config: dict, selected_modules: List[str]):
        """
        Starts the analyzer loop in a background thread.

        analysis_config:
            {
                "file_path": str,
                "refresh_interval_ms": int,
                "row_limit": int
            }

        selected_modules:
            ["statistics", "clustering", ...]
        """
        if self._thread and self._thread.is_alive():
            self.log_callback("AnalyzerLoop is already running.")
            return

        file_path = analysis_config.get("file_path")
        if not file_path:
            self.log_callback("No file path specified for analysis.")
            return

        self.selected_modules = selected_modules
        self.reader = TelemetryReader(
            file_path=file_path,
            row_limit=analysis_config.get("row_limit", 10_000),
        )

        self.refresh_interval = analysis_config.get("refresh_interval_ms", 500) / 1000.0

        self.log_callback(f"Starting analysis on file: {file_path}")
        self.total_rows_processed = 0
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Requests the analyzer loop to stop and waits for thread to finish.
        """
        if not self._thread:
            return

        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self.log_callback("AnalyzerLoop stopped.")

    def set_modules(self, modules: List[str]):
        """
        Updates the list of selected modules at runtime.
        """
        self.selected_modules = modules
        self.log_callback(f"Modules updated: {modules}")

    # ---------------------------------------------------------
    # Internal Loop
    # ---------------------------------------------------------
    def _run_loop(self):
        """
        Background loop:
            - periodically reads new data
            - runs selected modules
            - updates GUI via callbacks
        """
        while not self._stop_event.is_set():
            try:
                # 1. Read new data
                new_data = self.reader.read_new_rows() if self.reader else None
                if new_data is None or new_data.empty:
                    # No new data; sleep and continue
                    self.progress_callback(0.0, "Waiting for new data...")
                    time.sleep(self.refresh_interval)
                    continue

                rows = len(new_data)
                self.total_rows_processed += rows
                self.log_callback(f"Read {rows} new rows (total: {self.total_rows_processed}).")

                # 2. Run selected modules
                results: Dict[str, Any] = {}
                health_updates: List[Dict[str, Any]] = []

                if "statistics" in self.selected_modules:
                    stats_result, stats_health = statistics.run(new_data)
                    results["time_series"] = stats_result
                    if stats_health:
                        health_updates.append(stats_health)

                if "clustering" in self.selected_modules:
                    cluster_result, cluster_health = clustering.run(new_data)
                    results["clustering"] = cluster_result
                    if cluster_health:
                        health_updates.append(cluster_health)

                if "forecasting" in self.selected_modules:
                    forecast_result, forecast_health = forecasting.run(new_data)
                    results["forecasting"] = forecast_result
                    if forecast_health:
                        health_updates.append(forecast_health)

                if "nlp" in self.selected_modules:
                    nlp_result, nlp_health = nlp.run(new_data)
                    results["nlp"] = nlp_result
                    if nlp_health:
                        health_updates.append(nlp_health)

                if "deep_learning" in self.selected_modules:
                    dl_result, dl_health = deep_learning.run(new_data)
                    results["deep_learning"] = dl_result
                    if dl_health:
                        health_updates.append(dl_health)

                if "xai" in self.selected_modules:
                    xai_result, xai_health = xai.run(new_data)
                    results["xai"] = xai_result
                    if xai_health:
                        health_updates.append(xai_health)

                # 3. Push visualization updates
                if results:
                    self.visualization_callback(results)

                # 4. Aggregate health
                if health_updates:
                    # Simple aggregation: worst status wins
                    aggregated = self._aggregate_health(health_updates)
                    self.health_callback(aggregated)

                # 5. Update progress (activity-based, since total is unknown)
                if not hasattr(self, "_activity_percent"):
                    self._activity_percent = 0.0

                # Each batch nudges the bar forward up to 100%
                self._activity_percent = min(100.0, self._activity_percent + 1.0)

                self.progress_callback(
                    self._activity_percent,
                    f"Processed {self.total_rows_processed} rows."
                )

                # 6. Sleep until next refresh
                time.sleep(self.refresh_interval)

            except Exception as e:
                self.log_callback(f"Error in AnalyzerLoop: {e}")
                self.health_callback({"status": "Error", "message": str(e)})
                time.sleep(self.refresh_interval)

    # ---------------------------------------------------------
    # Health Aggregation
    # ---------------------------------------------------------
    def _aggregate_health(self, health_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates multiple health dicts into a single summary.

        Priority: Error > Warning > OK
        """
        priority = {"OK": 0, "Warning": 1, "Error": 2}
        best_status = "OK"
        messages = []

        for h in health_list:
            status = h.get("status", "OK")
            msg = h.get("message", "")
            if msg:
                messages.append(msg)
            if priority.get(status, 0) > priority.get(best_status, 0):
                best_status = status

        return {
            "status": best_status,
            "message": " | ".join(messages) if messages else "",
        }