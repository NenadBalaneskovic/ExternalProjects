# -*- coding: utf-8 -*-

"""
================================================================================
PROJECT: AI Router & Hallucination Shield
MODULE:  gui/widgets.py
ROLE:    Custom specialized UI components, featuring the reactive quantum 
         hallucination status gauge panel and high-contrast telemetry readouts.
================================================================================
"""

import logging
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame
from PyQt6.QtCore import Qt, pyqtProperty

log = logging.getLogger("CustomWidgets")


class QuantumHallucinationShieldWidget(QFrame):
    """
    A custom telemetric visual indicator that shifts its background matrix,
    borders, and text alerts reactively depending on the evaluated H-Score.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        
        # Initialize internal baseline styles mimicking production Github theme
        self.setObjectName("HallucinationShield")
        self._set_base_layout()
        self.reset_to_initial_state()

    def _set_base_layout(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)
        
        # Upper structural title indicator
        title = QLabel("H-SCORE STATUS")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #8b949e; font-size: 11px; font-weight: bold; letter-spacing: 1px; border: none;")
        layout.addWidget(title)
        
        # Main central alert readout container
        self.status_label = QLabel("[ 0.00 - INITIAL ]")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #c9d1d9; font-size: 16px; font-weight: bold; border: none; padding: 10px 0px;")
        layout.addWidget(self.status_label)

    def reset_to_initial_state(self):
        """Restores the visual anchor to a neutral, waiting operational phase."""
        self.status_label.setText("[ 0.00 - STANDBY ]")
        self.setStyleSheet("""
            QFrame#HallucinationShield {
                background-color: #161b22;
                border: 2px solid #30363d;
                border-radius: 8px;
            }
        """)

    def update_shield_state(self, h_score: float, is_definitive: bool):
        """
        Dynamically shifts layout stylesheets and alerts depending on nonlinear 
        boundary thresholds matching our post-generation telemetry profiles.
        """
        # Scenario A: Secure Phase Space Path (Low-Drift Vector Basin)
        if h_score < 0.3:
            status_text = f"[ {h_score:.2f} - SECURE ✓ ]"
            bg_color = "#1f4728"      # Deep Dark Green
            border_color = "#238636"  # GitHub Green Accent
            text_color = "#58a6ff"    # High-contrast highlight blue
            
        # Scenario B: Stochastic Blur Transition (Moderate Drift / Warning Vector)
        elif h_score < 0.6:
            status_text = f"[ {h_score:.2f} - STOCHASTIC BLUR ⚠ ]"
            bg_color = "#4d3d0f"      # Deep Muted Amber
            border_color = "#d29922"  # GitHub Warning Yellow Accent
            text_color = "#f0e6d2"    # Pale clear cream text
            
        # Scenario C: Critical Disruption Matrix (Hyperbolical Hallucination Escape)
        else:
            status_text = f"[ {h_score:.2f} - CRITICAL DRIFT 🚨 ]"
            if is_definitive:
                status_text = f"[ {h_score:.2f} - DEFINITIVE HALLUCINATION 🚨 ]"
            bg_color = "#5c1d1a"      # Deep Blood Red
            border_color = "#f85149"  # GitHub Alert Red Accent
            text_color = "#ffffff"    # Pure stark white for maximum emergency contrast
            
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"color: {text_color}; font-size: 15px; font-weight: bold; border: none;")
        
        # Enforce structural border animation shifts via style injection
        self.setStyleSheet(f"""
            QFrame#HallucinationShield {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 8px;
            }}
        """)
        log.info(f"Visual protection shield updated to vector state amplitude: [{h_score:.4f}]")


class MetricsTelemetryGridPanel(QFrame):
    """
    A unified data grid layout displaying micro and macro parameters 
    extracted from the core non-linear dynamic metrics modules.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 10px;
            }
            QLabel {
                color: #c9d1d9;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                border: none;
            }
            QLabel#MetricHeader {
                color: #8b949e;
                font-family: inherit;
                font-weight: bold;
                font-size: 11px;
            }
        """)
        self._initialize_grid_layout()

    def _initialize_grid_layout(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        header = QLabel("📈 METRICS TELEMETRY GRID")
        header.setObjectName("MetricHeader")
        layout.addWidget(header)
        
        # Instantiate sub-labels tracking individual state parameters
        self.lbl_sqd = QLabel("• SQD-Drift:   0.0000")
        self.lbl_rg = QLabel("• RG-Flow:     0.0000")
        self.lbl_topo = QLabel("• Topo-Drift:  0.0000")
        self.lbl_blur = QLabel("• Phase Blur:  0.0000")
        
        layout.addWidget(self.lbl_sqd)
        layout.addWidget(self.lbl_rg)
        layout.addWidget(self.lbl_topo)
        layout.addWidget(self.lbl_blur)
        layout.addStretch()

    def render_metrics_package(self, raw_metrics: dict, mean_drift_x: float, trace_covariance_y: float):
        """Refreshes text readouts with sub-millisecond precision on the view matrix."""
        self.lbl_sqd.setText(f"• SQD-Drift:   {raw_metrics['sqd']:.4f}")
        self.lbl_rg.setText(f"• RG-Flow:     {raw_metrics['rg_instability']:.4f}")
        self.lbl_topo.setText(f"• Topo-Drift:  {raw_metrics['graph_entropy_drift']:.4f}")
        self.lbl_blur.setText(f"• Phase Blur:  {trace_covariance_y:.4f}")
        log.debug("Telemetry metrics panel fields populated with current time slice parameters.")

    def reset_grid(self):
        """Clears the tabular panel back to system initialization defaults."""
        self.lbl_sqd.setText("• SQD-Drift:   0.0000")
        self.lbl_rg.setText("• RG-Flow:     0.0000")
        self.lbl_topo.setText("• Topo-Drift:  0.0000")
        self.lbl_blur.setText("• Phase Blur:  0.0000")


log.info("Production custom telemetry widgets compiled successfully.")
