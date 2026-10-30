# analyzer/gui/module_panel.py

from PySide6.QtWidgets import (
    QWidget,
    QGroupBox,
    QVBoxLayout,
    QCheckBox,
)
from PySide6.QtCore import Signal


class ModulePanel(QWidget):
    """
    Module selection panel for the Telemetry Analyzer.

    Responsibilities:
        - Display toggles for each analysis module
        - Allow user to enable/disable modules dynamically
        - Emit a signal when module selection changes

    Signals:
        modules_changed -> emitted with a list of selected module names

    Methods:
        get_selected_modules() -> list[str]
            Returns the list of currently enabled modules
    """

    modules_changed = Signal(list)

    def __init__(self):
        super().__init__()
        self._build_ui()

    # ---------------------------------------------------------
    # UI Construction
    # ---------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()

        group = QGroupBox("Analysis Modules")
        group_layout = QVBoxLayout()

        # Define available modules
        self.module_checkboxes = {
            "statistics": QCheckBox("Statistics (rolling means, FFT)"),
            "clustering": QCheckBox("Clustering (KMeans, DBSCAN)"),
            "forecasting": QCheckBox("Forecasting (SARIMAX, LSTM)"),
            "nlp": QCheckBox("NLP (logs, topic modeling)"),
            "deep_learning": QCheckBox("Deep Learning (autoencoder anomaly detection)"),
            "xai": QCheckBox("Explainability (SHAP, Captum)"),
        }

        # Add checkboxes to layout
        for name, checkbox in self.module_checkboxes.items():
            checkbox.stateChanged.connect(self._emit_change)
            group_layout.addWidget(checkbox)

        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()

        self.setLayout(layout)

    # ---------------------------------------------------------
    # Signal emission
    # ---------------------------------------------------------
    def _emit_change(self):
        """
        Emits the modules_changed signal whenever a checkbox is toggled.
        """
        self.modules_changed.emit(self.get_selected_modules())

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def get_selected_modules(self) -> list:
        """
        Returns a list of module names that are currently enabled.
        """
        return [
            name
            for name, checkbox in self.module_checkboxes.items()
            if checkbox.isChecked()
        ]