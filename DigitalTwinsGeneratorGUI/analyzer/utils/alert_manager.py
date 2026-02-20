# analyzer/utils/alert_manager.py

from typing import Dict, Any, List
from datetime import datetime


class AlertManager:
    """
    Centralized alert formatter and storage for the Telemetry Analyzer.

    Responsibilities:
        - Receive raw alert dictionaries from AlertListener
        - Normalize and format them into human-readable messages
        - Store recent alerts for optional display or debugging
        - Provide helper methods for categorizing alert types

    Methods:
        add_alert(alert_dict) -> str
            Formats and stores an alert, returns formatted message

        get_recent_alerts(n=20) -> list[str]
            Returns the last n formatted alerts
    """

    def __init__(self, max_history: int = 200):
        self.max_history = max_history
        self.history: List[str] = []

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def add_alert(self, alert: Dict[str, Any]) -> str:
        """
        Accepts a raw alert dictionary from AlertListener and returns
        a formatted, human-readable message.

        Expected alert format:
            {
                "event": "chunk_written",
                "timestamp": "2026-02-17T10:42:00",
                "payload": { "rows": 10000 }
            }

        Returns:
            formatted_message: str
        """

        event = alert.get("event", "unknown_event")
        timestamp = alert.get("timestamp") or datetime.utcnow().isoformat()
        payload = alert.get("payload", {})

        # Build readable message
        payload_str = ", ".join(f"{k}={v}" for k, v in payload.items()) if payload else "no details"
        message = f"[ALERT] {event} at {timestamp} — {payload_str}"

        # Store in history
        self._store(message)

        return message

    def get_recent_alerts(self, n: int = 20) -> List[str]:
        """
        Returns the last n formatted alerts.
        """
        return self.history[-n:]

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    def _store(self, message: str):
        """
        Stores a formatted alert message in the history buffer.
        """
        self.history.append(message)
        if len(self.history) > self.max_history:
            self.history.pop(0)