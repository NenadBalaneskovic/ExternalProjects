# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  core/database.py
ROLE:    Persistent local SQLite storage engine managing thread-safe access
         to web session configurations, context states, and trajectory metrics.
================================================================================
"""

import sqlite3
import logging
from datetime import datetime

log = logging.getLogger("DatabasePersistence")


class SessionPersistenceManager:
    """
    Manages a thread-safe local SQLite database storing active browser 
    session states, thread identifiers, historical text, and trajectory metrics.
    """
    def __init__(self, db_path: str = "storage.db"):
        self.db_path = db_path
        self._initialize_schema()

    def _get_connection(self):
        """
        Returns a pristine connection instance to the local database file.
        Enforces key-based column access via sqlite3.Row configuration.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enables column extraction via string keys
        return conn

    def _initialize_schema(self):
        """Constructs tables for relational session definitions and telemetry tracking."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table A: Active Web Sessions, Threads, and Project URLs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_url_id TEXT UNIQUE NOT NULL,
                    agent_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    chat_type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Table B: Trajectory Physical Telemetry and Post-Generation Logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    prompt_text TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    h_score REAL NOT NULL,
                    mean_drift_x REAL NOT NULL,
                    trace_covariance_y REAL NOT NULL,
                    sqd_val REAL NOT NULL,
                    rg_val REAL NOT NULL,
                    topo_drift REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            conn.commit()
            log.info("SQLite persistence tables initialized and schema locked successfully.")

    def register_new_web_session(self, url_id: str, agent: str, title: str, chat_type: str) -> int:
        """
        Registers a fresh thread URL string or stream descriptor inside the local cache.
        If the unique URL hash already exists, it retrieves the matching key index.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO sessions (session_url_id, agent_name, title, chat_type, created_at) 
                    VALUES (?, ?, ?, ?, ?)
                """, (url_id, agent, title, chat_type, now))
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Session URL already cached in a prior interaction; return existing key match
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM sessions WHERE session_url_id = ?", (url_id,))
                row = cursor.fetchone()
                return row["id"] if row else -1

    def log_trajectory_step(self, session_id: int, step: int, prompt: str, text: str, metrics: dict):
        """Appends a precise evaluation log slice to the historical metrics log grid."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics_log (
                    session_id, step, prompt_text, response_text, h_score, 
                    mean_drift_x, trace_covariance_y, sqd_val, rg_val, topo_drift
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, step, prompt, text, metrics["h_score"],
                metrics["mean_drift_x"], metrics["trace_covariance_y"],
                metrics["raw_metrics"]["sqd"], metrics["raw_metrics"]["rg_instability"],
                metrics["raw_metrics"]["graph_entropy_drift"]
            ))
            conn.commit()
            log.info(f"Trajectory step {step} successfully logged for Session ID {session_id}.")

    def fetch_all_active_sessions(self) -> list:
        """
        Extracts complete historical logs mapped directly to the history table interface.
        Retrieves the latest execution metrics state slice for each session profile.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.id, s.title, s.agent_name, s.chat_type, 
                       COALESCE(m.h_score, 0.0) as last_h_score
                FROM sessions s
                LEFT JOIN metrics_log m ON m.session_id = s.id 
                AND m.step = (SELECT MAX(step) FROM metrics_log WHERE session_id = s.id)
                ORDER BY s.id DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def find_session_id_for_prompt(self, prompt: str):
        """
        Returns the session_id of the most recent session whose log already
        contains this EXACT prompt text, or None if no session has ever been
        asked this exact prompt. Lets the caller continue accumulating steps
        under one session (for a clean metrics-over-time trajectory) when the
        user resubmits the same prompt, instead of always starting fresh.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id FROM metrics_log
                WHERE prompt_text = ?
                ORDER BY log_id DESC
                LIMIT 1
            """, (prompt,))
            row = cursor.fetchone()
            return row["session_id"] if row else None

    def delete_sessions(self, session_ids: list):
        """
        Deletes one or more sessions and their metrics_log rows. Deletes
        metrics_log explicitly first: SQLite does not enforce FOREIGN KEY ...
        ON DELETE CASCADE unless "PRAGMA foreign_keys = ON" was set on the
        connection, which this app does not currently do, so relying on the
        schema's CASCADE clause alone would silently leave orphaned rows.
        """
        if not session_ids:
            return
        placeholders = ",".join("?" for _ in session_ids)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM metrics_log WHERE session_id IN ({placeholders})", session_ids)
            cursor.execute(f"DELETE FROM sessions WHERE id IN ({placeholders})", session_ids)
            conn.commit()
            log.info(f"Deleted {len(session_ids)} session(s) and their trajectory logs: {session_ids}")

    def delete_all_sessions(self):
        """Wipes every session and every logged trajectory step."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM metrics_log")
            cursor.execute("DELETE FROM sessions")
            conn.commit()
            log.info("All sessions and trajectory logs deleted.")

    def fetch_latest_log_for_session(self, session_id: int):
        """
        Returns the most recent metrics_log row for one session (prompt,
        response, and full diagnostic), or None if the session has no logged
        interaction yet. Used to actually restore a session's last exchange
        into the main window instead of just switching to an empty view.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT prompt_text, response_text, h_score, mean_drift_x,
                       trace_covariance_y, sqd_val, rg_val, topo_drift
                FROM metrics_log
                WHERE session_id = ?
                ORDER BY step DESC
                LIMIT 1
            """, (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def fetch_metrics_for_session(self, session_id: int) -> list:
        """
        Returns the full step-by-step trajectory telemetry for one session,
        ordered by step, for use in the metrics plot visualization.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT step, h_score, mean_drift_x, trace_covariance_y,
                       sqd_val, rg_val, topo_drift
                FROM metrics_log
                WHERE session_id = ?
                ORDER BY step ASC
            """, (session_id,))
            return [dict(row) for row in cursor.fetchall()]


log.info("SessionPersistenceManager successfully mounted to application context.")