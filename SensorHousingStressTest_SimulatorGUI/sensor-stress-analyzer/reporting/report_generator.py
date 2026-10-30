# reporting/report_generator.py
"""
Report generator for Sensor Stress Analyzer.
Creates governance-ready PDF reports including metadata, results, plots, and XAI summaries.
"""

import os
from fpdf import FPDF
import numpy as np
from config import REPORT_FILENAME, APP_NAME, APP_VERSION
from .xai_explainer import explain_results


def _sanitize_for_pdf(obj):
    """Convert NumPy arrays and other complex objects to PDF-safe strings."""
    if isinstance(obj, np.ndarray):
        return str(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_pdf(v) for k, v in obj.items()}
    if isinstance(obj, str):
        # Replace unsupported Unicode characters with safe ASCII equivalents
        return (
            obj.replace("\u2014", "-")  # em-dash → hyphen
               .replace("\u2013", "-")  # en-dash → hyphen
               .replace("\u2022", "*")  # bullet → asterisk
        )
    return str(obj)


class ReportGenerator:
    """Generates PDF reports for Sensor Stress Analyzer."""

    def __init__(self, results: dict):
        self.results = results
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def _add_header(self):
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 10, f"{APP_NAME} v{APP_VERSION}", ln=True, align="C")
        self.pdf.ln(10)

    def _add_section_title(self, title: str):
        self.pdf.set_font("Arial", "B", 12)
        safe_title = _sanitize_for_pdf(title)
        self.pdf.cell(0, 10, safe_title, ln=True)
        self.pdf.ln(5)

    def _add_text(self, text: str):
        self.pdf.set_font("Arial", "", 10)
        safe_text = _sanitize_for_pdf(text)
        self.pdf.multi_cell(0, 8, safe_text)
        self.pdf.ln(5)

    def generate(self, filename: str = REPORT_FILENAME):
        """Generate the PDF report with results, plots, and XAI summary."""
        self.pdf.add_page()
        self._add_header()

        # --- Results Section ---
        self._add_section_title("Simulation Results")
        safe_results = _sanitize_for_pdf(self.results)
        for key, value in safe_results.items():
            self._add_text(f"{key}: {value}")

        # --- Plots Section ---
        self._add_section_title("Simulation Plots")
        # Expect plots saved earlier as files
        for plot_file in ["rod_plot.png", "fem_plot.png", "fem_stress_heatmap.png", "fem_heat_heatmap.png"]:
            if os.path.exists(plot_file):
                self.pdf.image(plot_file, w=100)
                self.pdf.ln(5)

        # --- XAI Summary Section ---
        self._add_section_title("Explainable AI Summary")
        summary = explain_results(self.results)
        self._add_text(summary)

        # --- Save PDF ---
        self.pdf.output(filename)
        return filename
