# Sensor Stress Analyzer

A modular, narratable simulation and reporting platform for analyzing stress and heat propagation in n‑gonal sensor housings.  
Designed for consulting demos, engineering education, and governance‑ready reporting.

---

## 🚀 Features

- **GUI (PyQt5)**  
  - Sidebar controls for polygon corners, force, heat, and FEM toggle  
  - Visualization panel with placeholders for rod and FEM results  
  - Event handlers connecting inputs to simulation logic

- **Simulation Layer**  
  - `rod_analysis.py`: Analytical sinusoidal deflection model  
  - `fem_solver.py`: FEM‑style stress and heat propagation stubs  
  - `mesh_generator.py`: Polygon mesh creation for FEM visualization  
  - `solver_utils.py`: Acceleration hooks, benchmarking, and validation

- **Visualization Layer**  
  - `rod_plotter.py`: Polygon rod‑structure visualization  
  - `fem_plotter.py`: Stress/heat map visualization  
  - `color_maps.py`: Custom colormaps for stress and heat

- **Reporting Layer**  
  - `report_generator.py`: Narratable PDF reports with inputs, results, and figures  
  - `xai_explainer.py`: Explainability commentary for stakeholder clarity  
  - `export_utils.py`: Save results, text reports, and figures

- **Testing**  
  - Unit tests for rod analysis, FEM solver, and GUI interactions (`pytest`, `pytest‑qt`)

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/sensor-stress-analyzer.git
cd sensor-stress-analyzer
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the application:

```bash
python main.py
```

- Adjust **polygon corners (n)**, **force**, and **heat** using sliders.  
- Toggle **FEM mode** for deeper analysis.  
- View results in the visualization panel.  
- Export narratable PDF reports with explainability commentary.

---

## 🧪 Testing

Run unit tests with:

```bash
pytest
```

GUI tests require `pytest-qt`.

---

## 📂 Project Structure

```
sensor-stress-analyzer/
│
├── main.py
├── config.py
│
├── gui/
│   ├── main_window.py
│   ├── sidebar_controls.py
│   ├── visualization_panel.py
│   └── event_handlers.py
│
├── simulation/
│   ├── rod_analysis.py
│   ├── fem_solver.py
│   ├── mesh_generator.py
│   └── solver_utils.py
│
├── visualization/
│   ├── rod_plotter.py
│   ├── fem_plotter.py
│   └── color_maps.py
│
├── reporting/
│   ├── report_generator.py
│   ├── xai_explainer.py
│   └── export_utils.py
│
└── tests/
    ├── test_rod_analysis.py
    ├── test_fem_solver.py
    └── test_gui_interactions.py
```

---

## 📖 Notes

- **Extendability**: Replace placeholder math with real FEM routines (NumPy/SciPy).  
- **Visualization**: Upgrade placeholders with matplotlib (2D) or PyVista (3D).  
- **Reporting**: Embed figures directly into PDFs for consulting‑ready deliverables.  
- **Governance**: Narratable outputs ensure transparency and stakeholder clarity.

---

## 👤 Author

Developed by **Nenad Balaneskovic**  
Focused on modular workflow design, explainable AI, and governance‑ready reporting.

```
