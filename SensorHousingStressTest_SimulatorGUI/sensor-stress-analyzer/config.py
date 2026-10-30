# config.py
"""
Configuration file for Sensor Stress Analyzer.
Defines standardized filenames, constants, and global settings.
"""

# --- Report and export filenames ---
REPORT_FILENAME = "sensor_stress_report.pdf"
RESULTS_FILENAME = "sensor_stress_results.json"
TEXT_REPORT_FILENAME = "sensor_stress_summary.txt"
CSV_EXPORT_FILENAME = "sensor_stress_results.csv"

# --- Default simulation parameters ---
DEFAULT_N = 12          # Default polygon corners
DEFAULT_FORCE = 50      # Default applied force [N]
DEFAULT_HEAT = 50       # Default applied heat [°C]

# --- FEM solver settings ---
MESH_RESOLUTION = 400   # Grid resolution for FEM heatmaps
PADDING = 0.15          # Padding around polygon for visualization
EPSILON = 1e-9          # Numerical stability constant

# --- Visualization settings ---
ROD_PLOT_FILENAME = "rod_plot.png"
FEM_LINE_PLOT_FILENAME = "fem_plot.png"
FEM_STRESS_HEATMAP_FILENAME = "fem_stress_heatmap.png"
FEM_HEAT_HEATMAP_FILENAME = "fem_heat_heatmap.png"
FEM_COMBINED_HEATMAP_FILENAME = "fem_heatmap.png"

# --- Logging ---
LOG_FILE = "sensor_stress_log.txt"

# --- Application metadata ---
APP_NAME = "Sensor Stress Analyzer"
APP_VERSION = "1.0.0"
