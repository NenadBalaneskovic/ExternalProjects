# generator/core/alert_socket.py

import json
import socket
import threading
from typing import Dict, Any


class AlertSocketClient:
    """
    Lightweight non-blocking client for sending JSON alerts
    from the Generator to the Analyzer.

    Alerts include:
        - "generator_started"
        - "chunk_written"
        - "generation_complete"
        - "file_size_limit_reached"
        - custom messages

    The Analyzer listens on (host, port) and receives these events.

    Parameters:
        host: str
            Target host (default: "127.0.0.1")
        port: int
            Target port (default: 5050)
        enabled: bool
            Whether socket alerts are enabled
    """

    def __init__(self, host: str, port: int, enabled: bool = True):
        self.host = host
        self.port = port
        self.enabled = enabled

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def send_alert(self, event: str, payload: Dict[str, Any] | None = None):
        """
        Sends an alert asynchronously to avoid blocking the generator loop.

        Args:
            event: str
                Name of the event (e.g., "chunk_written")
            payload: dict
                Additional data to send (optional)
        """
        if not self.enabled:
            return

        message = {
            "event": event,
            "payload": payload or {}
        }

        # Send in background thread
        thread = threading.Thread(
            target=self._send_message,
            args=(message,),
            daemon=True
        )
        thread.start()

    # ---------------------------------------------------------
    # Internal Socket Logic
    # ---------------------------------------------------------
    def _send_message(self, message: Dict[str, Any]):
        """
        Sends a single JSON message over TCP.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)  # avoid blocking
                s.connect((self.host, self.port))
                s.sendall(json.dumps(message).encode("utf-8"))
        except Exception:
            # Silent fail — alerts are optional and must not break generation
            pass