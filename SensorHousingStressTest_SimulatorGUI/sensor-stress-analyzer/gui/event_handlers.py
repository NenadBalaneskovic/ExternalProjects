# gui/event_handlers.py

import os
from PyQt5.QtCore import QObject
from simulation.rod_analysis import run_rod_analysis
from simulation.fem_solver import run_fem_analysis
from reporting.report_generator import ReportGenerator
from reporting.export_utils import save_results, save_text_report, save_csv
from config import REPORT_FILENAME, RESULTS_FILENAME, TEXT_REPORT_FILENAME


class EventHandlers(QObject):
    def __init__(self, sidebar, visualization):
        super().__init__()
        self.sidebar = sidebar
        self.visualization = visualization
        self.last_results = None

        # Connect sidebar controls to actions
        self.sidebar.run_button.clicked.connect(self.run_analysis)
        self.visualization.save_pdf_button.clicked.connect(self.save_pdf)
        self.visualization.save_data_button.clicked.connect(self.save_data)

    def log(self, message: str):
        """Forward log messages to the visualization panel."""
        if hasattr(self.visualization, "log_message"):
            self.visualization.log_message(message)
        else:
            print(message)  # fallback to console

    def run_analysis(self):
        n = self.sidebar.n_slider.value()
        force = self.sidebar.force_slider.value()
        heat = self.sidebar.heat_slider.value()
        use_fem = self.sidebar.fem_checkbox.isChecked()

        self.log(f"Running analysis: n={n}, force={force}, heat={heat}, FEM={use_fem}")

        # Always update rod plot
        self.visualization.update_rod_plot(n, force, heat)

        if use_fem:
            results = run_fem_analysis(n, force, heat)
            self.visualization.update_fem_plot(results["stress_map"], results["heat_map"])
            self.visualization.update_fem_heatmaps(
                results["n"], results["stress_map"], results["heat_map"]
            )
        else:
            results = run_rod_analysis(n, force, heat)

        # Update summary and log
        self.visualization.update_xai_summary(results)
        self.log("Simulation complete.")
        self.last_results = results

    def save_pdf(self):
        """Save the current simulation results to a PDF report."""
        if not self.last_results:
            self.log("No results available to save.")
            return

        try:
            report = ReportGenerator(self.last_results)
            filename = report.generate(REPORT_FILENAME)
            self.log(f"PDF report saved: {filename}")
        except Exception as e:
            self.log(f"Error saving PDF report: {e}")

    def save_data(self):
        """Save the current simulation results to JSON, TXT, and CSV files."""
        if not self.last_results:
            self.log("No results available to save.")
            return

        try:
            json_file = save_results(self.last_results)
            txt_file = save_text_report(self.last_results)
            csv_file = save_csv(self.last_results)
            self.log(f"Data files saved: {json_file}, {txt_file}, {csv_file}")
        except Exception as e:
            self.log(f"Error saving data files: {e}")
