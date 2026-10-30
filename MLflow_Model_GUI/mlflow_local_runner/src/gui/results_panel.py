# mlflow_local_runner/src/gui/results_panel.py
"""
results_panel.py – Anzeige der Ergebnisse eines MLflow-Runs
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout, QTextBrowser
)
from PySide6.QtCore import Qt

from utils.logger import get_logger


class ResultsPanel(QWidget):
    """
    Panel zur Anzeige der Ergebnisse eines abgeschlossenen Runs.
    """

    def __init__(self):
        super().__init__()

        self.logger = get_logger(__name__)
        self._init_ui()

    # ---------------------------------------------------------
    # UI INITIALISIERUNG
    # ---------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Run-Ergebnisse")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # -------------------------
        # Run-Informationen
        # -------------------------
        self.info_label = QLabel("Run-Informationen:")
        self.info_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(self.info_label)

        self.info_text = QTextBrowser()
        self.info_text.setOpenExternalLinks(False)
        self.info_text.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(self.info_text)

        # -------------------------
        # Metriken
        # -------------------------
        self.metrics_label = QLabel("Metriken:")
        self.metrics_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(self.metrics_label)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        layout.addWidget(self.metrics_text)

        # -------------------------
        # Modell
        # -------------------------
        self.model_label = QLabel("Modell:")
        self.model_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(self.model_label)

        self.model_text = QTextBrowser()
        self.model_text.setOpenExternalLinks(False)
        self.model_text.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(self.model_text)

        # -------------------------
        # MLflow-Links
        # -------------------------
        self.links_label = QLabel("MLflow-Links:")
        self.links_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(self.links_label)

        self.links_text = QTextBrowser()
        self.links_text.setOpenExternalLinks(True)
        self.links_text.setTextInteractionFlags(Qt.TextBrowserInteraction)
        layout.addWidget(self.links_text)

        # -------------------------
        # Buttons
        # -------------------------
        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("Ergebnisse löschen")
        btn_layout.addWidget(self.btn_clear)
        layout.addLayout(btn_layout)

        self.btn_clear.clicked.connect(self.clear_results)

        self.setLayout(layout)

    # ---------------------------------------------------------
    # ERGEBNISSE ANZEIGEN
    # ---------------------------------------------------------

    def display_results(self, results: dict):
        """
        Zeigt die Ergebnisse eines Runs an.
        """

        self.logger.info("Zeige Ergebnisse im ResultsPanel an.")

        # -------------------------
        # Run-Informationen
        # -------------------------
        project_name = results.get("project_name", "") or "(nicht gesetzt)"
        experiment_name = results.get("experiment_name", "") or "(nicht gesetzt)"
        run_name = results.get("run_name", "") or "(nicht gesetzt)"

        info_html = (
            f"<b>Projektname:</b> {project_name}<br>"
            f"<b>Experimentname:</b> {experiment_name}<br>"
            f"<b>Runname:</b> {run_name}<br>"
        )
        self.info_text.setHtml(info_html)

        # -------------------------
        # Metriken
        # -------------------------
        metrics = results.get("metrics", {})
        metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
        self.metrics_text.setPlainText(metrics_text)

        # -------------------------
        # Modellrepräsentation
        # -------------------------
        model_repr = results.get("model_repr", "Keine Modellinformationen verfügbar.")
        self.model_text.setPlainText(str(model_repr))

        # -------------------------
        # MLflow-Links
        # -------------------------
        links = results.get("mlflow_links", {})

        run_url = links.get("run_url")
        artifact_url = links.get("artifact_url")
        model_url = links.get("model_url")

        html = ""

        if run_url:
            html += f'<b>Run:</b> <a href="{run_url}">{run_url}</a><br><br>'

        if artifact_url:
            html += f'<b>Artefakte:</b> <a href="{artifact_url}">{artifact_url}</a><br><br>'

        if model_url:
            html += f'<b>Modell:</b> <a href="{model_url}">{model_url}</a><br><br>'

        self.links_text.setHtml(html)

    # ---------------------------------------------------------
    # ERGEBNISSE LÖSCHEN
    # ---------------------------------------------------------

    def clear_results(self):
        self.info_text.clear()
        self.metrics_text.clear()
        self.model_text.clear()
        self.links_text.clear()
        self.logger.info("Ergebnisse gelöscht.")
