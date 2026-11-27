# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  gui/auth_window.py
ROLE:    Simple API-key management dialogs. Replaces the former Chromium
         cookie-harvesting flow: the user pastes an official API key (from
         Google AI Studio / Groq Console / Mistral Console / Tavily) and it
         is persisted to the local .env file. No browser automation,
         no session-cookie capture.
================================================================================
"""

import os
import logging
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QLineEdit, QFrame, QMessageBox)
from PyQt6.QtCore import Qt

log = logging.getLogger("AuthWindowEngine")

# Target mapping correlating providers to their .env key name and where to
# obtain a free API key. Verify current signup/pricing details on each
# provider's own page before relying on them, as free-tier terms change.
PROVIDER_CONFIGS = {
    "gemini": {
        "name": "Google Gemini",
        "env_key": "GEMINI_API_KEY",
        "help_url": "https://aistudio.google.com/apikey",
    },
    "groq": {
        "name": "Groq",
        "env_key": "GROQ_API_KEY",
        "help_url": "https://console.groq.com/keys",
    },
    "mistral": {
        "name": "Mistral AI",
        "env_key": "MISTRAL_API_KEY",
        "help_url": "https://console.mistral.ai/api-keys",
    },
    "tavily": {
        "name": "Tavily",
        "env_key": "TAVILY_API_KEY",
        "help_url": "https://app.tavily.com",
    },
}


def _get_env_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, '.env')


def _read_env_lines(env_path: str) -> list:
    if not os.path.exists(env_path):
        return []
    with open(env_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def _write_env_value(env_key: str, value: str):
    """Persists a single KEY=value pair to the local .env file, preserving other lines."""
    env_path = _get_env_path()
    env_lines = _read_env_lines(env_path)

    updated = False
    try:
        with open(env_path, 'w', encoding='utf-8') as f:
            for line in env_lines:
                if line.startswith(env_key + "="):
                    f.write(f"{env_key}={value}\n")
                    updated = True
                else:
                    f.write(line)
            if not updated:
                f.write(f"{env_key}={value}\n")
    except OSError as e:
        log.error(f"Failed to write .env at [{env_path}]: {e}")
        raise

    log.info(f"Local storage verified. Environment synced for target: [{env_key}]")


def _env_has_value(env_key: str) -> bool:
    env_path = _get_env_path()
    if not os.path.exists(env_path):
        return False
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
    target = f"{env_key}="
    if target not in content:
        return False
    value = content.split(target, 1)[1].split('\n')[0].strip()
    return len(value) > 8  # a real key is always longer than this


class ApiKeyEntryDialog(QDialog):
    """A small modal for pasting a single provider's API key."""
    def __init__(self, provider_key: str, parent=None):
        super().__init__(parent)
        if provider_key not in PROVIDER_CONFIGS:
            raise ValueError(f"Invalid provider key passed to entry dialog: {provider_key}")

        self.config = PROVIDER_CONFIGS[provider_key]
        self.provider_key = provider_key

        self.setWindowTitle(f"API-Key eintragen - {self.config['name']}")
        self.resize(480, 180)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        info = QLabel(
            f"\U0001F511 Trage deinen kostenlosen API-Key fuer {self.config['name']} ein.\n"
            f"Der Key wird ausschliesslich lokal in der .env-Datei gespeichert."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #58a6ff; font-weight: bold;")
        layout.addWidget(info)

        link = QLabel(f'<a href="{self.config["help_url"]}" style="color:#58a6ff;">Key hier erzeugen: {self.config["help_url"]}</a>')
        link.setOpenExternalLinks(True)
        layout.addWidget(link)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText(f"{self.config['env_key']}=...")
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.key_input)

        show_row = QHBoxLayout()
        self.btn_toggle_visibility = QPushButton("Anzeigen")
        self.btn_toggle_visibility.setCheckable(True)
        self.btn_toggle_visibility.toggled.connect(self._toggle_echo_mode)
        show_row.addWidget(self.btn_toggle_visibility)
        show_row.addStretch()
        layout.addLayout(show_row)

        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("Speichern")
        self.btn_cancel = QPushButton("Abbrechen")
        self.btn_save.clicked.connect(self._save_key)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_save)
        layout.addLayout(btn_row)

    def _toggle_echo_mode(self, checked: bool):
        self.key_input.setEchoMode(QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password)

    def _save_key(self):
        value = self.key_input.text().strip()
        if not value:
            QMessageBox.warning(self, "Eingabe leer", "Bitte einen gueltigen API-Key eintragen.")
            return
        try:
            _write_env_value(self.config["env_key"], value)
        except OSError as e:
            QMessageBox.critical(self, "Speichern fehlgeschlagen", str(e))
            return
        self.accept()


class AccountStatusCenterDialog(QDialog):
    """
    Overview panel showing which providers have a key stored locally, with a
    button per row to set/update it.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API-KEY SYNC CENTER")
        self.resize(550, 320)
        self.setModal(True)

        self._build_status_dashboard()
        self.refresh_account_statuses()

    def _build_status_dashboard(self):
        layout = QVBoxLayout(self)

        header = QLabel("Konfigurierte API-Provider:")
        header.setStyleSheet("color: #c9d1d9; font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)

        self.rows_widgets = {}

        for key, conf in PROVIDER_CONFIGS.items():
            row_frame = QFrame()
            row_frame.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; margin-bottom: 5px;")
            row_layout = QHBoxLayout(row_frame)
            row_layout.setContentsMargins(10, 8, 10, 8)

            label_name = QLabel(conf["name"])
            label_name.setStyleSheet("color: #c9d1d9; font-weight: bold; font-size: 13px; border: none;")

            label_status = QLabel("Checking...")
            label_status.setStyleSheet("font-weight: bold; border: none;")

            btn_sync = QPushButton("Key setzen")
            btn_sync.setStyleSheet("padding: 4px 8px; font-size: 11px;")
            btn_sync.clicked.connect(lambda checked, k=key: self._launch_key_entry(k))

            row_layout.addWidget(label_name)
            row_layout.addStretch()
            row_layout.addWidget(label_status)
            row_layout.addWidget(btn_sync)

            layout.addWidget(row_frame)
            self.rows_widgets[key] = {"status_lbl": label_status, "sync_btn": btn_sync}

        btn_close = QPushButton("Close Center")
        btn_close.clicked.connect(self.accept)
        layout.addSpacing(15)
        layout.addWidget(btn_close)

    def refresh_account_statuses(self):
        """Checks the local .env file to see which providers already have a key."""
        for key, conf in PROVIDER_CONFIGS.items():
            active = _env_has_value(conf["env_key"])
            self._update_row_ui(key, active=active)

    def _update_row_ui(self, key: str, active: bool):
        lbl = self.rows_widgets[key]["status_lbl"]
        btn = self.rows_widgets[key]["sync_btn"]

        if active:
            lbl.setText("\u2713 ACTIVE / READY")
            lbl.setStyleSheet("color: #238636; font-weight: bold; border: none;")  # GitHub Green
            btn.setText("Key erneuern")
        else:
            lbl.setText("\u2715 KEIN KEY")
            lbl.setStyleSheet("color: #f85149; font-weight: bold; border: none;")  # GitHub Red
            btn.setText("Key setzen")

    def _launch_key_entry(self, provider_key: str):
        dialog = ApiKeyEntryDialog(provider_key, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            log.info(f"API key stored for provider: [{provider_key}]")
            self.refresh_account_statuses()