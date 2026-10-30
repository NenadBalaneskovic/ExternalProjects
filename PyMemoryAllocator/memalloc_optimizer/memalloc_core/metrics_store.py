"""
metrics_store.py

Responsible for:
- Persistently storing profiling metrics (runtime, memory, speedup)
- Providing fast analytical queries for the GUI and plots
- Using DuckDB for local, zero-config scientific storage
- Exporting per-run artifacts (CSV, JSON, logs, flamegraphs)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import duckdb
import time
import json
import csv
import cProfile
import pstats
from io import StringIO


# ============================================================
# Data structures
# ============================================================

@dataclass
class MetricRecord:
    """Single profiling metric entry."""
    timestamp: str
    script_hash: str
    runtime_seconds: float
    peak_memory_mb: float
    optimized: bool
    speedup: float
    strategy_summary: str


# ============================================================
# Metrics Store
# ============================================================

class MetricsStore:
    """
    Stores and retrieves profiling metrics using DuckDB.
    Also exports per-run artifacts (CSV, JSON, logs, flamegraphs).
    """

    EXPECTED_COLUMNS = [
        "timestamp",
        "script_hash",
        "runtime_seconds",
        "peak_memory_mb",
        "optimized",
        "speedup",
        "strategy_summary",
    ]

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = duckdb.connect(str(self.db_path))
        self._create_or_migrate_table()

    # --------------------------------------------------------
    # Table creation + auto-migration
    # --------------------------------------------------------

    def _create_or_migrate_table(self):
        """Ensure metrics table exists and matches expected schema."""

        tables = self.conn.execute("SELECT table_name FROM duckdb_tables;").fetchall()
        table_names = {t[0] for t in tables}

        if "metrics" not in table_names:
            self._create_table()
            return

        columns = self.conn.execute("PRAGMA table_info('metrics');").fetchall()
        existing_cols = [col[1] for col in columns]

        if existing_cols != self.EXPECTED_COLUMNS:
            print(">>> MIGRATING metrics TABLE (schema mismatch detected) <<<")
            self.conn.execute("DROP TABLE metrics;")
            self._create_table()

    def _create_table(self):
        """Create metrics table with correct schema."""
        self.conn.execute("""
            CREATE TABLE metrics (
                timestamp VARCHAR,
                script_hash VARCHAR,
                runtime_seconds DOUBLE,
                peak_memory_mb DOUBLE,
                optimized BOOLEAN,
                speedup DOUBLE,
                strategy_summary VARCHAR
            );
        """)

    # --------------------------------------------------------
    # Insert metrics
    # --------------------------------------------------------

    def insert_metric(
        self,
        script_hash: str,
        runtime_seconds: float,
        peak_memory_mb: float,
        optimized: bool,
        speedup: float,
        strategy_summary: str = ""
    ):
        """Insert a metric entry."""

        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        self.conn.execute("""
            INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?);
        """, [
            ts,
            script_hash,
            runtime_seconds,
            peak_memory_mb,
            optimized,
            speedup,
            strategy_summary
        ])

    # --------------------------------------------------------
    # Retrieve metrics
    # --------------------------------------------------------

    def get_all_metrics(self, script_hash: Optional[str] = None) -> List[Dict]:
        """Return all metrics or only metrics for a specific script."""

        if script_hash:
            query = """
                SELECT *
                FROM metrics
                WHERE script_hash = ?
                ORDER BY timestamp;
            """
            result = self.conn.execute(query, [script_hash]).fetchall()
        else:
            query = "SELECT * FROM metrics ORDER BY timestamp;"
            result = self.conn.execute(query).fetchall()

        return [
            {
                "timestamp": row[0],
                "script_hash": row[1],
                "runtime_seconds": row[2],
                "peak_memory_mb": row[3],
                "optimized": bool(row[4]),
                "speedup": row[5],
                "strategy_summary": row[6],
            }
            for row in result
        ]

    def get_metrics_by_hash(self, script_hash: str):
        return [m for m in self.get_all_metrics() if m["script_hash"] == script_hash]

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    def create_record(
        self,
        script_hash: str,
        runtime_seconds: float,
        peak_memory_mb: float,
        optimized: bool,
        speedup: float,
        strategy_summary: str = ""
    ) -> MetricRecord:
        """Convenience method to create a MetricRecord with timestamp."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        return MetricRecord(
            timestamp=ts,
            script_hash=script_hash,
            runtime_seconds=runtime_seconds,
            peak_memory_mb=peak_memory_mb,
            optimized=optimized,
            speedup=speedup,
            strategy_summary=strategy_summary,
        )

    # ============================================================
    # Export utilities (CSV, JSON, flamegraph, logs)
    # ============================================================

    def _safe_filename(self, record: MetricRecord, suffix: str) -> Path:
        """Generate a safe filename for per-run artifacts."""
        ts = record.timestamp.replace(":", "-").replace(" ", "_")
        tag = "optimized" if record.optimized else "baseline"
        return Path(f"{ts}_{tag}{suffix}")

    def export_json(self, record: MetricRecord, metrics_dir: Path):
        """Export a single metric record as JSON."""
        metrics_dir.mkdir(exist_ok=True)
        filename = metrics_dir / self._safe_filename(record, ".json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(record.__dict__, f, indent=4)
        return filename

    def export_csv(self, record: MetricRecord, metrics_dir: Path):
        """Export a single metric record as CSV."""
        metrics_dir.mkdir(exist_ok=True)
        filename = metrics_dir / self._safe_filename(record, ".csv")
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(record.__dict__.keys())
            writer.writerow(record.__dict__.values())
        return filename

    def export_log(self, stdout: str, stderr: str, metrics_dir: Path, record: MetricRecord):
        """Store stdout/stderr logs for each run."""
        metrics_dir.mkdir(exist_ok=True)
        filename = metrics_dir / self._safe_filename(record, ".log")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("STDOUT:\n")
            f.write(stdout)
            f.write("\n\nSTDERR:\n")
            f.write(stderr)
        return filename

    def export_flamegraph(self, func, metrics_dir: Path, record: MetricRecord):
        """Generate a flamegraph-like SVG for the executed function."""
        metrics_dir.mkdir(exist_ok=True)
        filename = metrics_dir / self._safe_filename(record, "_flamegraph.svg")

        profiler = cProfile.Profile()
        profiler.enable()
        func()  # run the function being profiled
        profiler.disable()

        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats()

        with open(filename, "w", encoding="utf-8") as f:
            f.write("<pre>\n")
            f.write(s.getvalue())
            f.write("</pre>\n")

        return filename
