# analyzer/core/alert_listener.py

import socket
import threading
import json
from typing import Callable, Optional


class AlertListener:
    """
    Background TCP listener for alerts sent by the Generator.

    Responsibilities:
        - Open a TCP socket on (host, port)
        - Accept incoming connections from AlertSocketClient
        - Parse JSON messages
        - Forward alerts to the GUI via callback
        - Run safely in a background thread

    Alerts typically look like:
        {
            "event": "chunk_written",
            "payload": {"rows": 10000}
        }
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5050,
        enabled: bool = True,
        alert_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.host = host
        self.port = port
        self.enabled = enabled
        self.alert_callback = alert_callback

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._socket: Optional[socket.socket] = None

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def start(self):
        """
        Starts the alert listener in a background thread.
        """
        if not self.enabled:
            return

        if self._thread and self._thread.is_alive():
            return  # already running

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stops the listener and closes the socket.
        """
        self._stop_event.set()

        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass

        if self._thread:
            self._thread.join(timeout=1.0)

    # ---------------------------------------------------------
    # Internal Loop
    # ---------------------------------------------------------
    def _run(self):
        """
        Main loop:
            - Bind to socket
            - Accept connections
            - Read JSON messages
            - Forward to callback
        """
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.host, self.port))
            self._socket.listen(5)
        except Exception:
            # If binding fails, silently disable listener
            return

        while not self._stop_event.is_set():
            try:
                self._socket.settimeout(0.5)
                try:
                    conn, _ = self._socket.accept()
                except socket.timeout:
                    continue  # loop again

                with conn:
                    data = conn.recv(4096)
                    if not data:
                        continue

                    try:
                        message = json.loads(data.decode("utf-8"))
                    except Exception:
                        continue  # ignore malformed JSON

                    if self.alert_callback:
                        self.alert_callback(message)

            except Exception:
                # Listener must never crash
                continue