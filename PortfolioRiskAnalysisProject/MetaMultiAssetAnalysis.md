# 1. 🚀 Project Introduction: Meta Multi-Asset Management

## Objective  
Meta and functional programming paradigms offer significant flexibility in designing robust and flexible GUIs via PyQt5 capable of automatically
adapting their widget structure to continuously changing data set schemas. Especially in the course of multi asset management tasks such
flexibilities promise higher efficiencies in simultaneously evaluating numerous portfolio structures comprised of diverse asset classes. 
Therefore, multi asset management may be regarded as an adequate playground for testing highly adaptable GUI designs and their UX performance 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/PortfolioRiskAnalysisProject/MetaMultiAssetAnalysis.md#7--references) 1 - 3 below). 

### 🎯 **Primary Aim**

1. Design a data analysis (pythonic) project regarding portfolio, risk and volatility analysis used within the framework of a multi-asset 
management company.   
2. Generate a large synthetic data set and design a pyqt gui which uses modern physical and statistical methods and accomodates its widget 
structure to feature changes within a csv data set by means of functional and meta programming paradigms.
 
### 🧠 **Tasks**

#### 🧠 Project Vision: “FIN QuantLab” (working title)

A Python-based interactive platform for simulating, analyzing, and visualizing portfolio risk and volatility dynamics using synthetic data 
and real-world modeling techniques. The GUI adapts to data structure changes and supports exploratory workflows through functional and metaprogramming.

#### 🧱 Core Modules

##### 1. 📊 Synthetic Data Generator
- **Purpose**: Simulate realistic multi-asset time series with configurable statistical properties.
- **Features**:
  - Correlated GBM or mean-reverting processes (Ornstein-Uhlenbeck, Heston)
  - Regime-switching volatility (Markov-switching models)
  - Macroeconomic factor injection (e.g., interest rate shocks)
- **Implementation**:
  - `numpy`, `scipy`, `pandas`, `statsmodels`
  - Configurable YAML/JSON schema for scenario design

##### 2. 📈 Portfolio & Risk Engine
- **Purpose**: Compute portfolio metrics and risk decomposition
- **Features**:
  - Rolling volatility, VaR (historical, parametric, Monte Carlo)
  - Marginal/Component risk contributions
  - PCA-based factor risk
  - Drawdown analysis
- **Implementation**:
  - Functional pipelines using `toolz` or `functools`
  - Plug-and-play risk models via decorators/metaclasses

##### 3. 🧠 Physics-Inspired Analytics
- **Purpose**: Apply physical/statistical mechanics to market dynamics
- **Ideas**:
  - Entropy of return distributions
  - Fractal dimension / Hurst exponent
  - Kalman-filtered state estimation
  - Langevin dynamics for volatility clustering

##### 4. 🧪 PyQt GUI: “QuantCanvas”
- **Purpose**: Interactive dashboard for data exploration
- **Features**:
  - Dynamic widget generation based on CSV schema
  - Drag-and-drop asset selection
  - Real-time charting with `pyqtgraph` or `matplotlib`
  - Theme-aware layout (light/dark mode)
- **Implementation**:
  - Meta-programming to auto-generate widgets from data types
  - Signal-slot architecture for reactive updates

##### 5. 🧬 Functional & Meta Programming Layer
- **Purpose**: Ensure modularity, extensibility, and introspection
- **Techniques**:
  - Decorators for registering models, metrics, and views
  - Dynamic class generation for GUI components
  - Currying and partial application for pipeline construction
  - Plugin system for user-defined analytics

##### 6. 📁 Data I/O & Persistence
- **Purpose**: Seamless integration with CSV, Excel, and APIs
- **Features**:
  - Schema-aware CSV loader with validation
  - Auto-refresh on file change (watchdog)
  - Export to PDF/Excel reports

## 🧭 Next Steps

1. **Define the synthetic data schema**: What asset classes, time horizons, and statistical properties?
2. **Sketch the GUI layout**: What are the core panels (e.g., asset selector, risk metrics, plots)?
3. **Choose the functional/metaprogramming patterns**: Registry decorators? Dynamic class factories?
4. **Decide on the physics/statistics crossover**: Which models are most relevant to the portfolio analysis framework?

# 2. 🔐 GUI and algorithmic concepts

## 2.1 Initial remarks

This is a project blending GUI design, statistical modeling, functional programming, and financial analytics into a modular, adaptive system. 
Let’s architect the PyQt GUI structure that integrates all six components outlined above, with a focus on extensibility and responsiveness to data schema changes.

### 🧱 GUI Architecture Overview: “FIN QuantLab”

| Module | Purpose | GUI Component |
|--------|---------|----------------|
| 1. Synthetic Data Generator | Create large-scale simulated datasets | Feature selector + schema builder + generator panel |
| 2. Portfolio & Risk Engine | Compute metrics like VaR, volatility, drawdown | Risk dashboard + metric selector |
| 3. Physics-Inspired Analytics | Apply Kalman filters, entropy, fractals | Analytics panel with model dropdown |
| 4. PyQt GUI | Interactive, schema-aware interface | Main window with dynamic widgets |
| 5. Functional & Meta Programming | Modular, introspective design | Decorator-driven widget registry |
| 6. Data I/O & Persistence | Load/save CSV, Excel, auto-refresh | File loader + watcher + export panel |


### 🖼️ GUI Layout Sketch

```
MainWindow (QMainWindow)
├── MenuBar
│   ├── File (Load CSV, Save, Export)
│   └── Tools (Generate Data, Run Analysis)
├── CentralWidget (QWidget)
│   └── QVBoxLayout
│       ├── SchemaPanel (QGroupBox)
│       │   └── FeatureSelector (QListWidget + QCheckBoxes)
│       ├── GeneratorPanel (QGroupBox)
│       │   └── Controls (QSpinBox for rows, QComboBox for distributions)
│       ├── RiskPanel (QTabWidget)
│       │   ├── VolatilityTab
│       │   ├── VaRTab
│       │   └── DrawdownTab
│       ├── PhysicsPanel (QTabWidget)
│       │   ├── KalmanTab
│       │   ├── EntropyTab
│       │   └── FractalTab
│       └── PlotPanel (QGraphicsView or pyqtgraph)
└── StatusBar
```

### 🧪 Synthetic Data Generator (100,000+ rows)

```python
def generate_synthetic_data(schema: dict, n_rows: int = 100_000) -> pd.DataFrame:
    import numpy as np, pandas as pd
    data = {}
    for col, dist in schema.items():
        if dist == "normal":
            data[col] = np.random.normal(loc=0, scale=1, size=n_rows)
        elif dist == "uniform":
            data[col] = np.random.uniform(low=0, high=1, size=n_rows)
        elif dist == "gbm":
            dt = 1/252
            mu, sigma = 0.1, 0.2
            prices = [100]
            for _ in range(n_rows - 1):
                prices.append(prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal()))
            data[col] = prices
    return pd.DataFrame(data)
```

GUI integration:
- FeatureSelector: lets user choose column names and distributions
- GeneratorPanel: triggers generation and updates PlotPanel

### 🧬 Functional & Meta Programming

Use decorators to register analytics modules:

```python
ANALYTICS_REGISTRY = {}

def register_analysis(name):
    def wrapper(func):
        ANALYTICS_REGISTRY[name] = func
        return func
    return wrapper

@register_analysis("Kalman Filter")
def kalman_analysis(df): ...
```

GUI dynamically populates dropdowns from `ANALYTICS_REGISTRY`.

### 🔄 Schema-Aware Widget Generation

When a CSV is loaded:
- Parse column types
- Auto-generate widgets (sliders, dropdowns) for each column
- Use metaclasses or factory functions to instantiate widgets

```python
def create_widget_for_column(col_name, dtype):
    if dtype == 'float':
        return QDoubleSpinBox()
    elif dtype == 'int':
        return QSpinBox()
    elif dtype == 'object':
        return QComboBox()
```

### 📁 Data I/O & Watchdog

- Use `QFileDialog` for loading/saving
- Use `watchdog` to monitor CSV changes and refresh GUI
- Export analysis results to Excel/PDF

## 2.2 Detailed GUI and algorithmic sketch  

Here's a detailed sketch of the GUI layout for our “FIN QuantLab” project, integrating all six aspects into a modular, extensible PyQt interface. 
This structure is designed to be schema-aware, reactive to CSV changes, and friendly to functional and metaprogramming extensions.

---

## 🖼️ GUI Layout Sketch: FIN QuantLab

```
MainWindow (QMainWindow)
├── MenuBar
│   ├── File
│   │   ├── Load CSV
│   │   ├── Save Dataset
│   │   ├── Export Report
│   └── Tools
│       ├── Generate Synthetic Data
│       ├── Run Risk Analysis
│       ├── Apply Physics Models
│       └── Refresh Widgets
├── CentralWidget (QWidget)
│   └── QVBoxLayout
│       ├── DataSchemaPanel (QGroupBox)  ← Aspect 1: Synthetic Data Generator
│       │   ├── FeatureSelector (QListWidget + QCheckBoxes)
│       │   ├── DistributionSelector (QComboBox per feature)
│       │   └── GenerateButton (QPushButton)
│       ├── RiskAnalysisPanel (QTabWidget)  ← Aspect 2: Portfolio & Risk Engine
│       │   ├── VolatilityTab
│       │   │   ├── RollingWindowInput (QSpinBox)
│       │   │   └── VolPlot (pyqtgraph or matplotlib)
│       │   ├── VaRTab
│       │   │   ├── MethodSelector (QComboBox: Historical, Parametric, Monte Carlo)
│       │   │   └── VaRPlot
│       │   ├── DrawdownTab
│       │   │   └── DrawdownPlot
│       ├── PhysicsAnalyticsPanel (QTabWidget)  ← Aspect 3: Physics-Inspired Analytics
│       │   ├── KalmanTab
│       │   │   ├── StateSelector (QComboBox)
│       │   │   └── KalmanPlot
│       │   ├── EntropyTab
│       │   │   └── EntropyPlot
│       │   ├── FractalTab
│       │   │   └── HurstPlot
│       ├── DynamicWidgetPanel (QScrollArea)  ← Aspect 4: PyQt GUI + Aspect 5: Meta Programming
│       │   └── AutoGeneratedWidgets (created from CSV schema)
│       │       ├── Column1Widget (QDoubleSpinBox or QComboBox)
│       │       ├── Column2Widget (...)
│       │       └── ...
│       ├── DataIOPanel (QGroupBox)  ← Aspect 6: Data I/O & Persistence
│       │   ├── FilePathLabel (QLabel)
│       │   ├── RefreshButton (QPushButton)
│       │   ├── ExportFormatSelector (QComboBox: CSV, Excel, PDF)
│       │   └── ExportButton
│       └── PlotPanel (QGraphicsView or pyqtgraph)
│           └── InteractivePlotArea
└── StatusBar
    ├── MessageLabel
    └── ProgressBar
```

## 🔧 Functional Highlights

- **DynamicWidgetPanel**: Uses metaprogramming to generate widgets based on CSV column types.
- **DataSchemaPanel**: Allows users to define synthetic data schema and generate large datasets.
- **RiskAnalysisPanel**: Modular tabs for volatility, VaR, and drawdown — each pluggable via decorators.
- **PhysicsAnalyticsPanel**: Hosts advanced models like Kalman filters and entropy metrics.
- **DataIOPanel**: Handles file loading, schema parsing, auto-refresh, and export.

# 3. Main code (conceptual explanation)

Let’s dive into the inner workings of our updated **QuantLab GUI**, now enhanced with plugin discovery and YAML schema support. 
This isn’t just a GUI — it’s a modular, introspectable analytics platform built for extensibility, experimentation, and team-friendly workflows.

## 🧠 Architectural Overview

FIN QuantLab is composed of six tightly integrated modules:

| Module | Purpose |
|--------|---------|
| 📊 Synthetic Data Generator | Simulate multi-asset time series with configurable statistical properties |
| 📈 Portfolio & Risk Engine | Compute portfolio metrics and risk decomposition |
| 🧠 Physics-Inspired Analytics | Apply statistical mechanics to financial time series |
| 🧪 QuantCanvas GUI | Interactive dashboard for exploration and visualization |
| 🧬 Functional & Meta Programming Layer | Enable plugin architecture and introspection |
| 📁 Data I/O & Persistence | Load/export CSV, Excel, PDF, YAML schemas |

## 🧱 1. Synthetic Data Generator

### Models Supported:
- **GBM**: Correlated geometric Brownian motion
- **OU**: Ornstein-Uhlenbeck mean-reverting process
- **Heston**: Stochastic volatility with mean-reverting variance
- **Regime-Switching**: Volatility regimes driven by hidden Markov states

### YAML Integration:
- `load_yaml_schema(path)` → loads simulation parameters
- `generate_synthetic_data(**schema)` → dispatches to appropriate model
- `save_yaml_schema(schema, path)` → exports current config

This allows scenario design to be externalized and version-controlled.  

Let’s walk through the **synthetic data generator’s role** in the QuantLab GUI and what happens when one loads a synthetic CSV.

### 🧪 Synthetic Data Generator: Purpose & Output

#### 🔧 What it does:
- Generates **multi-asset time series** using models like GBM, OU, Heston, and Regime-Switching.
- Accepts parameters either programmatically or via a **YAML schema** (e.g., number of assets, steps, volatility).
- Returns a `pandas.DataFrame` with simulated price paths.

#### 📤 Output options:
- **In-memory**: Used directly inside the GUI for analysis and visualization.
- **Exportable**: You can save it as a **CSV, Excel, or PDF** via the GUI’s export button.

While the generator itself doesn’t automatically write to CSV, the GUI lets us export the generated data to CSV manually.

### 📥 What Happens After One Loads a Synthetic CSV

#### Step-by-step flow:

1. **File Selection**  
   You click “Load CSV” and select a synthetic dataset you previously exported or generated externally.

2. **Schema Validation**  
   The GUI runs `validate_schema(df)` to ensure:
   - No missing values
   - All columns are numeric (for risk/physics models)

3. **Schema Parsing**  
   It runs `parse_schema(df)` to classify each column as:
   - `numeric` → asset prices or returns
   - `categorical` → labels or regimes
   - `datetime` → timestamps (future-ready)

4. **Widget Generation**  
   Based on schema:
   - Numeric columns get metric selectors (mean, std, min, max)
   - Categorical columns get filter dropdowns
   - Date columns are reserved for time filters

5. **Asset Selection**  
   You choose assets via drag-and-drop from the `QListWidget`.

6. **Chart Rendering**  
   `update_chart()` plots selected assets using `pyqtgraph`.

7. **Analysis Ready**  
   You can now:
   - Apply risk models via `risk_pipeline()`
   - Apply physics models via `physics_pipeline()`
   - Export results or schema

#### 🧠 Why This Matters

This design makes the GUI:
- **Schema-aware**: Adapts to any dataset structure
- **Extensible**: New models can be added via plugins
- **Reproducible**: YAML schemas preserve simulation parameters
- **Interactive**: Immediate feedback via charts and widgets


Here’s a full example of a **YAML schema** for our synthetic data generator module. 
This schema defines parameters for a correlated GBM simulation with 6 assets and 1500 time steps — but it’s easily adaptable to other models.


### 📜 Example: `gbm_schema.yaml`

```yaml
model: gbm
n_assets: 6
n_steps: 1500
mu: 0.07
sigma: 0.25
corr: 0.6
```

### 🧩 How It Works in QuantLab

- **Import**: When you click “Import YAML” in the GUI, this schema is parsed and passed to `generate_synthetic_data(**schema)`.
- **Simulation**: The GBM model is selected and executed with the specified parameters.
- **Visualization**: The resulting DataFrame is validated, schema-parsed, and displayed in the GUI.
- **Export**: You can save the schema back to YAML using “Export YAML” for reproducibility.

### 🔄 Other Model Variants

You can swap out the `model` field and add relevant parameters:

#### Ornstein-Uhlenbeck
```yaml
model: ou
n_assets: 4
n_steps: 1000
theta: 0.2
mu: 0.0
sigma: 0.3
```

#### Heston
```yaml
model: heston
n_assets: 2
n_steps: 1200
mu: 0.05
kappa: 0.6
theta: 0.04
xi: 0.1
rho: -0.7
```

#### Regime-Switching
```yaml
model: regime
n_assets: 3
n_steps: 1000
regimes: 2
```

## 📈 2. Portfolio & Risk Engine

### Registered Models:
- Rolling volatility
- Historical, parametric, Monte Carlo VaR
- Marginal and component risk
- PCA factor risk
- Drawdown analysis

### Execution Flow:
- Decorators register models into `RISK_MODEL_REGISTRY`
- `risk_pipeline(df, model_name, **kwargs)` executes via `toolz.pipe` and `functools.partial`

This makes risk models plug-and-play and introspectable.

## 🧠 3. Physics-Inspired Analytics

### Registered Models:
- Entropy of return distributions
- Hurst exponent (fractal memory)
- Kalman filter (state smoothing)
- Langevin dynamics (volatility clustering)

### Execution Flow:
- Decorators register models into `PHYSICS_MODEL_REGISTRY`
- `physics_pipeline(df, model_name, **kwargs)` executes functionally

These models offer deeper insights into market microstructure and regime behavior.

## 🧪 4. QuantCanvas GUI

### Core Features:
- **Dynamic widget generation** from schema
- **Drag-and-drop asset selection**
- **Real-time charting** with `pyqtgraph`
- **Theme-aware layout** (light/dark toggle)
- **YAML import/export** for simulation scenarios

### Signal-Slot Architecture:
- `load_csv()` → validates schema, populates widgets
- `import_yaml_schema()` → loads synthetic data from YAML
- `generate_widgets()` → builds controls based on column types
- `update_chart()` → plots selected assets
- `apply_metric()` and `filter_category()` → respond to widget changes

This creates a reactive, schema-aware interface that adapts to any dataset.

## 🧬 5. Functional & Meta Programming Layer

### Techniques Used:
- **Decorators**: Register models and views
- **Registries**: Centralized dictionaries for introspection
- **Partial Application**: Parameterized pipelines
- **Plugin Discovery**: Loads external Python files from `plugins/`

### Plugin Loader:
```python
def discover_plugins(plugin_dir="plugins"):
    ...
```

This enables external teams to contribute analytics modules without modifying core code.

## 📁 6. Data I/O & Persistence

### Supported Formats:
- **CSV**: Standard tabular export
- **Excel**: For spreadsheet workflows
- **PDF**: Snapshot of tabular data
- **YAML**: Scenario configuration

### Export Flow:
- `export_dataset()` → saves current `df` in selected format
- `export_yaml_schema()` → saves simulation parameters

This ensures reproducibility and shareability across teams.

## 🔄 Full Workflow Example

1. **Import YAML schema** → loads simulation parameters
2. **Generate synthetic data** → using selected model
3. **Validate and parse schema** → auto-generates widgets
4. **Select assets and metrics** → triggers pipelines
5. **Apply risk/physics models** → via decorators and registries
6. **Visualize results** → in real-time chart
7. **Export dataset or schema** → for documentation or sharing


![PipelineWorkflowVisualization](https://github.com/NenadBalaneskovic/ExternalProjects/blob/0e6d95292313f2a9aeefc6c7d37b7a04c50532d8/PortfolioRiskAnalysisProject/Fig2.png)

The gui also stores two data csv files: here’s exactly what those two auto-saved CSVs contain and how they relate to our GUI:

### 📁 1. `simulation_YYYYMMDD_HHMMSS.csv`

#### ✅ Contents:
- This file contains the **raw synthetic asset data** generated from our YAML simulation block.
- Each column is an asset (e.g., `Asset_1`, `Asset_2`, …)
- Each row is a time step (e.g., 0 to 1499 for `n_steps: 1500`)
- Values are **simulated prices** from models like GBM, OU, Heston, etc.

#### ✅ Example:
| Asset_1 | Asset_2 | Asset_3 | … |
|---------|---------|---------|----|
| 100.00  | 100.00  | 100.00  | … |
| 100.12  | 99.87   | 100.23  | … |
| …       | …       | …       | … |

### 📁 2. `analysis_YYYYMMDD_HHMMSS.csv`

#### ✅ Contents:
- This file contains the **result of the analysis pipeline**, e.g., Rolling Volatility.
- Same columns as the simulation file (assets), but values are **volatility estimates** or other metrics.
- First few rows may be `NaN` due to rolling window.

#### ✅ Example:
| Asset_1 | Asset_2 | Asset_3 | … |
|---------|---------|---------|----|
| NaN     | NaN     | NaN     | … |
| 0.0123  | 0.0118  | 0.0132  | … |
| …       | …       | …       | … |

### ✅ Can They Be Loaded into the GUI?

**Yes**  
We can load either CSV using the **“Load CSV”** button in the GUI:

- If we load the **simulation CSV**, it will behave like fresh raw data.
- If we load the **analysis CSV**, it will behave like a post-analysis dataset — ready for charting and metric inspection.

Just make sure:
- The CSV has numeric columns only
- No missing values if you want to run schema validation (or disable that check temporarily)


---

# 🧠 4. Modularized Pythonic implementation  

### Run instructions

1. Download the folder 📁 [PortfolioRiskAnalysisProject](https://github.com/NenadBalaneskovic/ExternalProjects/tree/74f9b9ee01972a3c9ca1106aedb61629184c5cab/PortfolioRiskAnalysisProject)
 which has the following structure:
   <img src="https://github.com/NenadBalaneskovic/ExternalProjects/blob/a7e1ae86a65ad34bcefc541b707de58bd5bb8a07/PortfolioRiskAnalysisProject/Fig3.PNG" width="400" height="200"/>
2. Run the py file "__quantlab_launcher.py__" in VS Code.
3. Load one of the yaml schemas into the GUI.

## 4.1. quantlab_launcher.py

````python
from PyQt5.QtWidgets import QApplication
from main_gui import QuantCanvas

if __name__ == "__main__":
    print("Launching QuantCanvas...")
    app = QApplication([])
    window = QuantCanvas()
    window.show()
    app.exec_()
	````


## 4.2. main_gui.py

````python
import sys
import pandas as pd
import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QListWidget, QListWidgetItem,
    QFileDialog, QCheckBox, QFrame, QStatusBar, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal

from plugin_loader import discover_plugins
from schema_io import validate_schema, parse_schema, load_yaml_schema, save_yaml_schema
from synthetic_models import generate_synthetic_data
from risk_models import risk_pipeline
from physics_models import physics_pipeline


class QuantCanvas(QMainWindow):
    theme_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FIN QuantLab — QuantCanvas")
        self.resize(1400, 900)
        self.df = None
        self.raw_df = None
        self.schema = {}
        self.simulation_schema = {}

        discover_plugins()

        central = QWidget()
        self.layout = QVBoxLayout()
        central.setLayout(self.layout)
        self.setCentralWidget(central)

        controls = QHBoxLayout()
        load_btn = QPushButton("Load CSV")
        load_btn.clicked.connect(self.load_csv)
        import_yaml_btn = QPushButton("Import YAML")
        import_yaml_btn.clicked.connect(self.import_yaml_schema)
        export_yaml_btn = QPushButton("Export YAML")
        export_yaml_btn.clicked.connect(self.export_yaml_schema)
        self.theme_toggle = QCheckBox("Dark Mode")
        self.theme_toggle.stateChanged.connect(self.toggle_theme)
        controls.addWidget(load_btn)
        controls.addWidget(import_yaml_btn)
        controls.addWidget(export_yaml_btn)
        controls.addWidget(self.theme_toggle)
        self.layout.addLayout(controls)

        self.asset_list = QListWidget()
        self.asset_list.setDragEnabled(True)
        self.asset_list.setSelectionMode(QListWidget.MultiSelection)
        self.asset_list.setFixedHeight(100)
        self.layout.addWidget(QLabel("Assets"))
        self.layout.addWidget(self.asset_list)
        self.asset_list.itemSelectionChanged.connect(self.update_chart)

        self.widget_panel = QFrame()
        self.widget_layout = QVBoxLayout()
        self.widget_panel.setLayout(self.widget_layout)
        self.layout.addWidget(QLabel("Schema Widgets"))
        self.layout.addWidget(self.widget_panel)

        self.chart = pg.PlotWidget()
        self.chart.setTitle("Real-Time Chart")
        self.chart.setLabel("left", "Price")
        self.chart.setLabel("bottom", "Time", units="s")
        self.chart.addLegend(offset=(80, 20), anchor=(0, 0))
        self.layout.addWidget(self.chart)

        status_bar = QStatusBar()
        status_bar.addWidget(QLabel("Ready"))
        self.progress = QProgressBar()
        self.progress.setValue(0)
        status_bar.addPermanentWidget(self.progress)
        self.setStatusBar(status_bar)

    def toggle_theme(self, state):
        if state == Qt.Checked:
            self.setStyleSheet("background-color: #2b2b2b; color: white;")
        else:
            self.setStyleSheet("")

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if path:
            self.progress.setValue(10)
            self.df = pd.read_csv(path, parse_dates=True)
            self.raw_df = self.df.copy()
            try:
                validate_schema(self.df)
            except Exception as e:
                print(f"Schema validation failed: {e}")
                return
            self.schema = parse_schema(self.df)
            self.populate_assets()
            self.generate_widgets()
            self.update_chart()
            self.progress.setValue(100)

    def import_yaml_schema(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import YAML", "", "YAML Files (*.yaml *.yml)")
        if not path:
            return

        self.progress.setValue(5)
        schema = load_yaml_schema(path)

        sim_config = schema.get("simulation", {})
        self.simulation_schema = sim_config
        try:
            self.df = generate_synthetic_data(**sim_config)
            self.raw_df = self.df.copy()
            self.autosave_csv(self.df, label="simulation")
        except Exception as e:
            print(f"Simulation failed: {e}")
            return

        self.progress.setValue(25)
        try:
            validate_schema(self.df)
        except Exception as e:
            print(f"Schema validation failed: {e}")
            return

        self.schema = parse_schema(self.df)
        self.populate_assets()
        self.generate_widgets()
        self.update_chart()
        self.progress.setValue(60)

        analysis_config = schema.get("analysis", {})
        model_type = analysis_config.get("type", "risk")
        model_name = analysis_config.get("model")

        if model_name:
            try:
                if model_type == "risk":
                    returns = np.log(self.df / self.df.shift(1)).dropna()
                    result = risk_pipeline(returns, model_name)
                elif model_type == "physics":
                    result = physics_pipeline(self.df, model_name)
                else:
                    print(f"Unknown analysis type: {model_type}")
                    return

                print(f"{model_type.capitalize()} analysis result for {model_name}:\n", result.head())
                if isinstance(result, pd.DataFrame):
                    self.df = result
                    self.schema = parse_schema(self.df)
                    self.populate_assets()
                    self.update_chart()
                    self.autosave_csv(self.df, label="analysis")
                self.progress.setValue(100)
            except Exception as e:
                print(f"Analysis failed: {e}")

    def export_yaml_schema(self):
        if not self.simulation_schema:
            print("No schema to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export YAML", "", "YAML Files (*.yaml *.yml)")
        if path:
            save_yaml_schema(self.simulation_schema, path)
            print(f"Exported YAML schema to {path}")

    def populate_assets(self):
        self.asset_list.clear()
        self.asset_list.blockSignals(True)  # prevent premature triggers
        for col in self.df.columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsDragEnabled)
            self.asset_list.addItem(item)
        self.asset_list.blockSignals(False)

        # ✅ Explicitly select all items after population
        for i in range(self.asset_list.count()):
            self.asset_list.item(i).setSelected(True)

    def generate_widgets(self):
        # Clear existing widgets
        for i in reversed(range(self.widget_layout.count())):
            widget = self.widget_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.metric_labels = {}  # Store value labels per asset

        for col, dtype in self.schema.items():
            label = QLabel(f"{col} ({dtype})")
            label.setStyleSheet("font-weight: bold;")
            self.widget_layout.addWidget(label)

            if dtype == "numeric":
                combo = QComboBox()
                combo.addItems(["mean", "std", "min", "max"])
                combo.currentIndexChanged.connect(lambda _, c=col, w=combo: self.apply_metric(c, w))
                self.widget_layout.addWidget(combo)

                value_label = QLabel("Value: —")
                value_label.setStyleSheet("color: gray; margin-left: 10px;")
                self.widget_layout.addWidget(value_label)
                self.metric_labels[col] = value_label

            elif dtype == "categorical":
                combo = QComboBox()
                combo.addItems(self.df[col].astype(str).unique().tolist())
                combo.currentIndexChanged.connect(lambda _, c=col, w=combo: self.filter_category(c, w))
                self.widget_layout.addWidget(combo)

            elif dtype == "datetime":
                self.widget_layout.addWidget(QLabel("Time filter coming soon..."))

    def apply_metric(self, column, widget):
        metric = widget.currentText()
        if metric and column in self.df.columns:
            try:
                value = getattr(self.df[column], metric)()
                formatted = f"{value:.4f}" if isinstance(value, (int, float, np.number)) else str(value)
                print(f"{metric} of {column}: {formatted}")
                if column in self.metric_labels:
                    self.metric_labels[column].setText(f"Value: {formatted}")
            except Exception as e:
                print(f"Failed to compute {metric} for {column}: {e}")
                if column in self.metric_labels:
                    self.metric_labels[column].setText("Value: error")

    def filter_category(self, column, widget):
        value = widget.currentText()
        if value:
            filtered = self.df[self.df[column].astype(str) == value]
            print(f"Filtered {column} = {value}, {len(filtered)} rows")

    def autosave_csv(self, df, label="output"):
        import os
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}.csv"
        save_dir = "autosaves"
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)

        try:
            df.to_csv(path, index=False)
            print(f"Auto-saved {label} to {path}")
        except Exception as e:
            print(f"Failed to save {label}: {e}")

    def update_chart(self):
        if self.df is None or self.df.empty:
            print("No data to plot.")
            return

        selected_items = self.asset_list.selectedItems()
        selected_cols = [item.text() for item in selected_items if item.text() in self.df.columns]

        print("Selected columns:", selected_cols)
        print("Available columns:", self.df.columns.tolist())

        if not selected_cols:
            print("No assets selected.")
            return

        self.chart.clear()

        # Re-add legend with offset to avoid overlap
        self.chart.addLegend(offset=(80, 20), anchor=(0, 0))

        # Define color palette
        COLOR_PALETTE = [
            (255, 0, 0),      # Red
            (0, 128, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 165, 0),    # Orange
            (128, 0, 128),    # Purple
            (0, 206, 209),    # Turquoise
            (255, 192, 203),  # Pink
            (128, 128, 128),  # Gray
            (255, 215, 0),    # Gold
            (0, 0, 0)         # Black
        ]

        for i, col in enumerate(selected_cols):
            series = self.df[col].dropna()
            if not np.issubdtype(series.dtype, np.number):
                print(f"Skipping non-numeric column: {col}")
                continue
            if series.empty:
                print(f"No valid data to plot for {col}")
                continue
            x = np.arange(len(series))
            y = series.values
            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            pen = pg.mkPen(color=color, width=2)
            print(f"Plotting {col}: {len(y)} points with color {color}")
            self.chart.plot(x, y, pen=pen, name=col)

````

## 4.3. registries.py

````python

RISK_MODEL_REGISTRY = {}
PHYSICS_MODEL_REGISTRY = {}
VIEW_REGISTRY = {}
METRIC_REGISTRY = {}


def register_risk_model(name):
    def wrapper(func):
        RISK_MODEL_REGISTRY[name] = func
        return func
    return wrapper


def register_physics_model(name):
    def wrapper(func):
        PHYSICS_MODEL_REGISTRY[name] = func
        return func
    return wrapper


def register_view(name):
    def wrapper(cls):
        VIEW_REGISTRY[name] = cls
        return cls
    return wrapper


def register_metric(name):
    def wrapper(func):
        METRIC_REGISTRY[name] = func
        return func
    return wrapper

````

## 4.4. synthetic_models.py

````python

import numpy as np
import pandas as pd


def simulate_gbm(n_assets=5, n_steps=1000, mu=0.1, sigma=0.2, corr=0.5):
    dt = 1/252
    cov_matrix = np.full((n_assets, n_assets), corr)
    np.fill_diagonal(cov_matrix, 1.0)
    L = np.linalg.cholesky(cov_matrix)
    returns = np.random.normal(0, 1, (n_steps, n_assets)) @ L.T
    prices = np.full((n_steps, n_assets), 100.0)
    for t in range(1, n_steps):
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * returns[t])
    return pd.DataFrame(prices, 
                        columns=[f"Asset_{i+1}" for i in range(n_assets)])


def simulate_ou(n_assets=5, n_steps=1000, theta=0.15, mu=0.0, sigma=0.3):
    dt = 1/252
    x = np.zeros((n_steps, n_assets))
    for t in range(1, n_steps):
        x[t] = x[t-1] + theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(size=n_assets)
    return pd.DataFrame(x, columns=[f"OU_{i+1}" for i in range(n_assets)])


def simulate_heston(n_assets=1, n_steps=1000, mu=0.05, kappa=0.5, theta=0.04, xi=0.1, rho=-0.7):
    dt = 1/252
    S = np.full((n_steps, n_assets), 100.0)
    v = np.full((n_steps, n_assets), theta)
    for t in range(1, n_steps):
        z1 = np.random.normal(size=n_assets)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=n_assets)
        v[t] = np.abs(v[t-1] + kappa * (theta - v[t-1]) * dt + xi * np.sqrt(v[t-1] * dt) * z2)
        S[t] = S[t-1] * np.exp((mu - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1] * dt) * z1)
    return pd.DataFrame(S, columns=[f"Heston_{i+1}" for i in range(n_assets)])


def simulate_regime_switching(n_assets=3, n_steps=1000, regimes=2):
    dt = 1/252
    states = np.random.choice(regimes, size=n_steps)
    mu_vals = [0.05, -0.02]
    sigma_vals = [0.1, 0.3]
    prices = np.full((n_steps, n_assets), 100.0)
    for t in range(1, n_steps):
        regime = states[t]
        mu = mu_vals[regime]
        sigma = sigma_vals[regime]
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), size=n_assets)
        prices[t] = prices[t-1] * np.exp(returns)
    return pd.DataFrame(prices, columns=[f"RS_{i+1}" for i in range(n_assets)])


def generate_synthetic_data(model="gbm", n_assets=5, n_steps=1000, **kwargs):
    if model == "gbm":
        return simulate_gbm(n_assets, n_steps, **kwargs)
    elif model == "ou":
        return simulate_ou(n_assets, n_steps, **kwargs)
    elif model == "heston":
        return simulate_heston(n_assets, n_steps, **kwargs)
    elif model == "regime":
        return simulate_regime_switching(n_assets, n_steps, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model}")

````

## 4.5. risk_models.py

````python

import numpy as np
import pandas as pd
from scipy.stats import norm
from toolz import pipe
from functools import partial
from registries import register_risk_model, RISK_MODEL_REGISTRY

@register_risk_model("Rolling Volatility")
def compute_rolling_volatility(df, window=20):
    return df.rolling(window=window).std()

@register_risk_model("Historical VaR")
def compute_historical_var(df, confidence=0.95):
    return df.quantile(1 - confidence)

@register_risk_model("Parametric VaR")
def compute_parametric_var(df, confidence=0.95):
    z = norm.ppf(confidence)
    return df.mean() - z * df.std()

@register_risk_model("Monte Carlo VaR")
def compute_monte_carlo_var(df, confidence=0.95, n_sim=1000):
    sim_returns = np.random.normal(df.mean(), df.std(), (n_sim, len(df.columns)))
    return pd.Series(np.percentile(sim_returns, (1 - confidence) * 100, axis=0), index=df.columns)

@register_risk_model("Marginal Risk")
def compute_marginal_risk(df, weights=None):
    cov = df.cov()
    if weights is None:
        weights = np.ones(len(cov)) / len(cov)
    return pd.Series(cov @ weights, index=df.columns)

@register_risk_model("Component Risk")
def compute_component_risk(df, weights=None):
    cov = df.cov()
    if weights is None:
        weights = np.ones(len(cov)) / len(cov)
    total_var = weights.T @ cov @ weights
    marginal = cov @ weights
    return pd.Series(weights * marginal / total_var, index=df.columns)

@register_risk_model("PCA Factor Risk")
def compute_pca_risk(df, n_components=3):
    cov = df.cov()
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx][:n_components]
    return pd.Series(eigvals, name="Explained Variance")

@register_risk_model("Drawdown")
def compute_drawdown(df):
    cum_returns = (1 + df).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown

def risk_pipeline(df, model_name, **kwargs):
    model = RISK_MODEL_REGISTRY.get(model_name)
    if model:
        return pipe(df, partial(model, **kwargs))
    return None

````

## 4.6. physics_models.py

````python

import numpy as np
import pandas as pd
from toolz import pipe
from functools import partial
from registries import register_physics_model, PHYSICS_MODEL_REGISTRY

@register_physics_model("Entropy")
def compute_entropy(df):
    return -df.apply(lambda x: np.sum(x * np.log(np.abs(x) + 1e-9)), axis=0)

@register_physics_model("Hurst Exponent")
def compute_hurst(df):
    def hurst(ts):
        lags = range(2, 100)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    return pd.Series({col: hurst(df[col].dropna()) for col in df.columns})

@register_physics_model("Kalman Filter")
def apply_kalman_filter(df):
    return df.ewm(span=10).mean()

@register_physics_model("Langevin Dynamics")
def simulate_langevin(df, gamma=0.1, noise_scale=0.05):
    dt = 1/252
    x = df.copy()
    for col in x.columns:
        for t in range(1, len(x)):
            drift = -gamma * x[col].iloc[t-1]
            noise = noise_scale * np.random.normal()
            x.at[t, col] = x.at[t-1, col] + drift * dt + noise * np.sqrt(dt)
    return x

def physics_pipeline(df, model_name, **kwargs):
    model = PHYSICS_MODEL_REGISTRY.get(model_name)
    if model:
        return pipe(df, partial(model, **kwargs))
    return None

````

## 4.7. schema_io.py

````python

import yaml
import numpy as np

def validate_schema(df):
    if df.isnull().any().any():
        raise ValueError("Data contains missing values.")
    if not all(np.issubdtype(dtype, np.number) for dtype in df.dtypes):
        raise TypeError("All columns must be numeric.")
    return True

def parse_schema(df):
    schema = {}
    for col in df.columns:
        dtype = df[col].dtype
        if np.issubdtype(dtype, np.number):
            schema[col] = "numeric"
        elif np.issubdtype(dtype, np.datetime64):
            schema[col] = "datetime"
        else:
            schema[col] = "categorical"
    return schema

def load_yaml_schema(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml_schema(schema, path):
    with open(path, "w") as f:
        yaml.dump(schema, f)

````

## 4.8. plugin_loader.py

````python

import os
import importlib.util


def discover_plugins(plugin_dir="plugins"):
    if not os.path.exists(plugin_dir):
        return
    for filename in os.listdir(plugin_dir):
        if filename.endswith(".py"):
            path = os.path.join(plugin_dir, filename)
            spec = importlib.util.spec_from_file_location(filename[:-3], path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"Plugin {filename} failed to load: {e}")

````

## 4.9. yaml schemas

### 4.9.1 gbm_schema.yaml

````yaml

simulation:
  model: gbm
  n_assets: 6
  n_steps: 1500
  mu: 0.07
  sigma: 0.25
  corr: 0.6

analysis:
  type: risk         # or "physics"
  model: Rolling Volatility

````

### 4.9.2 heston_schema.yaml

````yaml

simulation:
 model: heston
 n_assets: 2
 n_steps: 1200
 mu: 0.05
 kappa: 0.6
 theta: 0.04
 xi: 0.1
 rho: -0.7

analysis:
  type: risk         # or "physics"
  model: Rolling Volatility

````

### 4.9.3 ou_schema.yaml

````yaml

simulation:
 model: ou
 n_assets: 4
 n_steps: 1000
 theta: 0.2
 mu: 0.0
 sigma: 0.3
 
analysis:
  type: risk         # or "physics"
  model: Rolling Volatility

````

### 4.9.4 regime_schema.yaml

````yaml

simulation:
 model: regime
 n_assets: 3
 n_steps: 1000
 regimes: 2
 
analysis:
  type: risk         # or "physics"
  model: Rolling Volatility

````

---

# 5. 🔗 Results and conclusions

## 5.1 📊 Start the GUI and load yaml files

1. Run the py file "__quantlab_launcher.py__" in VS Code.
2. Load one of the yaml schemas into the GUI.

![UsageVisualization](https://github.com/NenadBalaneskovic/ExternalProjects/blob/0b81460d27d9b78a23b99bb1e18447ac6df13532/PortfolioRiskAnalysisProject/Fig4.png)

## 5.2 🧠 Interpretation of results

- Does early engagement (e.g. first 24h activity) predict long-term retention?
- Can we identify churn predictors based on event sequences?
- What is the causal impact of tracker cost on downstream revenue?
![UsageVisualizationFunctionality](https://github.com/NenadBalaneskovic/ExternalProjects/blob/bc087bd52feb497ad21c0354b7a76427b4672139/PortfolioRiskAnalysisProject/Fig4_6.png)
- **Propensity score matching**: Compare similar users across campaigns.
- **Survival analysis**: Model time-to-churn.
- **Granger causality**: Test if one time series (e.g. installs) predicts another (e.g. revenue).
![GeneratedCsvFiles](https://github.com/NenadBalaneskovic/ExternalProjects/blob/bc087bd52feb497ad21c0354b7a76427b4672139/PortfolioRiskAnalysisProject/Fig4_5.PNG )

## 5.3 Technologies

This is a comprehensive list of all technologies, libraries, and packages that power our QuantCanvas GUI project, grouped by purpose and functionality:

### 🖥️ Core Technologies

| Category              | Tools & Packages                     | Purpose |
|----------------------|--------------------------------------|---------|
| **GUI Framework**     | `PyQt5`, `QtWidgets`, `QtCore`, `QtGui` | Building the interactive GUI, widgets, layout, signals |
| **Charting**          | `pyqtgraph`                          | Fast, interactive plotting of asset data |
| **Data Handling**     | `pandas`, `numpy`                    | DataFrames, numerical operations, time series analysis |
| **File I/O**          | `yaml`, `csv`, `os`, `datetime`      | Reading YAML schemas, autosaving CSVs, timestamping |
| **Simulation Models** | Custom modules: `simulate_gbm`, `simulate_ou`, `simulate_heston`, `simulate_regime_switching` | Generating synthetic asset paths |
| **Analysis Pipelines**| Custom modules: `risk_pipeline`, `physics_pipeline` | Applying volatility and physics-based analytics |
| **Schema Parsing**    | `parse_schema` (custom)              | Inferring column types for widget generation |
| **Environment**       | Python 3.x                           | Base language and runtime |

### 🧩 GUI Features Enabled by These Tools

- **Dynamic widget generation** based on schema
- **Real-time chart updates** with asset selection
- **Color-coded plots** with legend positioning
- **Metric inspection** (mean, std, min, max) per asset
- **YAML import/export** for reproducible simulations
- **Autosave of datasets** as timestamped CSVs
- **Responsive layout** with progress feedback

### 🛠️ Optional or Future Additions

| Feature               | Suggested Tools                      |
|----------------------|--------------------------------------|
| Chart overlays        | `pyqtgraph` with multiple plot layers |
| Export bundles        | `zipfile`, `json`, `matplotlib` (for snapshots) |
| Plugin architecture   | `importlib`, `inspect`, decorators   |
| Statistical modeling  | `scipy`, `statsmodels`, `sklearn`    |
| Regime diagnostics    | `hmmlearn`, `pymc`, `matplotlib`     |
| GUI enhancements      | `QtDesigner`, `QSettings`, `QSplitter` |

### 📦 Summary

Our project is a tightly integrated stack of:
- **PyQt5** for GUI
- **pyqtgraph** for visualization
- **pandas/numpy** for data
- **yaml/csv/os/datetime** for I/O and persistence
- **Custom simulation and analysis modules** for domain logic

It is modular, extensible, and built for performance and reproducibility, a perfect foundation for advanced financial analytics.

## 5.4 🧠 Final Review: QuantCanvas GUI — Design, Functionality, and Performance

Here is a comprehensive conclusion that captures the essence of our GUI’s architecture, its current capabilities, and its future trajectory:

### 🎨 Design Philosophy

QuantCanvas embodies a modular, user-centric design tailored for quantitative analytics. Its architecture reflects:

- **Schema-aware adaptability**: Widgets and controls dynamically respond to the structure of loaded data, whether from CSVs or YAML simulations.
- **Visual clarity**: Real-time charting with distinct color-coded assets, labeled axes, and responsive legends ensures intuitive interpretation.
- **Minimalist extensibility**: The GUI avoids clutter while remaining open to plugin-based expansion, making it ideal for interdisciplinary teams.

The layout balances control and feedback — users can simulate, analyze, inspect metrics, and visualize results without leaving the main interface.

### ⚙️ Functional Capabilities

QuantCanvas now supports a robust pipeline from data generation to analysis and visualization:

- **Simulation ingestion**: YAML-based models (GBM, OU, Heston, Regime Switching) are parsed and executed with reproducible parameters.
- **Analysis integration**: Risk and physics models (e.g., Rolling Volatility) are applied seamlessly, with results visualized and stored.
- **Interactive widgets**: Each asset has metric selectors (mean, std, min, max) with live value updates, enabling quick statistical inspection.
- **Charting engine**: PyQtGraph delivers fast, responsive plotting with dynamic asset selection, color differentiation, and labeled axes.
- **Autosave mechanism**: Every dataset — raw or analyzed — is archived as a timestamped CSV, ensuring reproducibility and traceability.

Together, these features create a closed-loop workflow for simulation, inspection, and export — ideal for research, teaching, or production-grade analytics.

### 🚀 Performance and Responsiveness

The GUI is optimized for:

- **Low-latency updates**: Asset selection triggers immediate chart refreshes, even with large datasets.
- **Memory efficiency**: NaNs are dropped before plotting, and only numeric columns are processed, minimizing overhead.
- **Scalability**: The architecture supports multiple assets, rolling metrics, and future model extensions without structural changes.

The use of PyQt and pandas ensures compatibility with high-performance backends and integration with scientific Python ecosystems.

### 🌱 Future Potential Improvements

QuantCanvas is already powerful, but its architecture invites further evolution:

#### 🧩 Functional Enhancements
- **Raw vs analyzed overlays**: Toggle or animate transitions between price and volatility views.
- **Plugin sandboxing**: Allow users to inject custom models or metrics via a plugin interface.
- **Export bundles**: Save YAML + CSV + chart snapshot as a reproducible analysis package.

#### 📊 Visualization Upgrades
- **Interactive legends**: Toggle asset visibility directly from the chart.
- **Zoom/pan/crosshair tools**: Enhance chart navigation and precision inspection.
- **Annotations**: Display metric values directly on the chart.

#### 🧠 Analytical Depth
- **Factor decomposition**: Integrate PCA or risk factor models.
- **Regime-aware diagnostics**: Highlight transitions in regime-switching models.
- **Batch testing**: Run multiple simulations and compare outputs statistically.

#### 🧪 Usability and Collaboration
- **Session logging**: Track user actions and autosaves for reproducibility.
- **Team mode**: Enable shared YAML libraries and collaborative analysis.
- **Settings panel**: Let users configure autosave paths, chart styles, and default models.

## 🏁 Final Thoughts

QuantCanvas is more than a GUI — it is a scalable, introspectable analytics platform. It reflects our vision for reproducible, 
interactive, and extensible financial modeling. With its modular backbone and reactive design, it is ready to evolve into a flagship 
tool for quantitative research and collaborative exploration.

---

# 6. 🌍 Glossary 

Here is a deep dive into each concept, organized by category and tailored to our analytics platform. 
These are foundational to both quantitative finance and statistical physics, and they’re all integrated into our QuantLab GUI.

Let’s also unpack the **formulas behind Marginal and Component Risk** in the context of portfolio variance. 
These are essential for understanding **risk attribution**, **risk budgeting**, and **optimization**.

## 🧮 6.1 Portfolio Variance Foundation

Let’s start with the total portfolio variance:

$\[
\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}
\]$

- $\(\mathbf{w}\)$: vector of asset weights  
- $\(\Sigma\)$: covariance matrix of asset returns  
- $\(\sigma_p^2\)$: total portfolio variance

## 📌 6.2 Marginal Risk Contribution (MRC)

### 🔍 Definition:
The **partial derivative** of portfolio risk with respect to asset \( i \)’s weight:

$\[
\text{MRC}_i = \frac{\partial \sigma_p}{\partial w_i}
\]$

### 🧠 Derivation:
Since $\(\sigma_p = \sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}\)$, we apply the chain rule:

$\[
\text{MRC}_i = \frac{1}{2 \sigma_p} \cdot \frac{\partial (\mathbf{w}^\top \Sigma \mathbf{w})}{\partial w_i}
= \frac{(\Sigma \mathbf{w})_i}{\sigma_p}
\]$

So the marginal risk of asset $\( i \)$ is proportional to its **covariance with the portfolio**.

## 📌 6.3 Component Risk Contribution (CRC)

### 🔍 Definition:
The **absolute contribution** of asset $\( i \)$ to total portfolio risk:

$\[
\text{CRC}_i = w_i \cdot \text{MRC}_i
\]$

### 🧠 Interpretation:
- Tells you **how much risk** each asset contributes in total.
- Sum of all CRCs equals total portfolio risk:
  $\[
  \sum_i \text{CRC}_i = \sigma_p
  \]$

### 🧠 Why It Matters

- **Risk Budgeting**: Allocate capital based on risk contribution.
- **Optimization**: Minimize risk subject to CRC constraints.
- **Diagnostics**: Identify over-contributing assets.

## 📈 6.4 Risk & Portfolio Analytics

### 🔁 Rolling Volatility
- **Definition**: Standard deviation of returns over a moving window (e.g. 20 days).
- **Purpose**: Captures time-varying risk; useful for detecting volatility clustering.
- **Formula**:  
  $\[
  \sigma_t = \sqrt{\frac{1}{N} \sum_{i=t-N+1}^{t} (r_i - \bar{r})^2}
  \]$

### 📉 Value at Risk (VaR) Variants

| Type | Description | Assumptions |
|------|-------------|-------------|
| **Historical VaR** | Quantile of past returns | Non-parametric |
| **Parametric VaR** | Uses mean and std deviation assuming normality | Gaussian |
| **Monte Carlo VaR** | Simulates returns from estimated distribution | Flexible, computationally intensive |

- **Formula (Parametric)**:  
  $\[
  \text{VaR}_{\alpha} = \mu - z_{\alpha} \cdot \sigma
  \]$
  where $\( z_{\alpha} \)$ is the z-score for confidence level $\( \alpha \)$

### 🧮 Marginal & Component Risk

- **Marginal Risk**: Sensitivity of portfolio risk to a small change in asset weight  
  $\[
  \frac{\partial \text{Risk}}{\partial w_i}
  \]$

- **Component Risk**: Contribution of each asset to total portfolio risk  
  $\[
  \text{CR}_i = w_i \cdot \text{Marginal}_i
  \]$

- **Use Case**: Risk budgeting, attribution, and optimization.

### 🧠 PCA Factor Risk

- **Definition**: Decomposes asset covariance into orthogonal factors.
- **Purpose**: Identifies dominant sources of risk.
- **Method**:
  - Eigen-decomposition of covariance matrix
  - Select top \( k \) components explaining most variance

- **Formula**:  
  $\[
  \Sigma = V \Lambda V^\top
  \]$
  where $\( \Lambda \)$ contains eigenvalues (factor variances)

### 📉 Drawdown Analysis

- **Definition**: Measures peak-to-trough decline in cumulative returns.
- **Formula**:  
  $\[
  \text{Drawdown}_t = \frac{P_t - \max(P_{1:t})}{\max(P_{1:t})}
  \]$

- **Use Case**: Stress testing, capital preservation, regime detection.

## 🧪 6.5 Physics-Inspired Analytics

### 🔥 Entropy of Return Distributions

- **Definition**: Measures disorder or unpredictability in returns.
- **Formula**:  
  $\[
  H(X) = -\sum p(x) \log p(x)
  \]$

- **Use Case**: Detecting regime shifts, complexity, or information content.

### 🌀 Hurst Exponent (Fractal Memory)

- **Definition**: Quantifies long-term memory in time series.
- **Interpretation**:
  - $\( H < 0.5 \)$: Mean-reverting
  - $\( H = 0.5 \)$: Random walk
  - $\( H > 0.5 \)$: Persistent trend

- **Method**: Rescaled range analysis or DFA.

### 📡 Kalman Filter (State Smoothing)

- **Definition**: Recursive estimator for hidden states in noisy systems.
- **Use Case**: Signal extraction, volatility smoothing, latent factor modeling.

- **Core Equations**:
  - Prediction:  
    $\[
    \hat{x}_{t|t-1} = A \hat{x}_{t-1|t-1}
    \]$
  - Update:  
    $\[
    \hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (y_t - H \hat{x}_{t|t-1})
    \]$

### 🌊 Langevin Dynamics (Volatility Clustering)

- **Definition**: Stochastic differential equation modeling noisy mean-reversion.
- **Equation**:  
  $\[
  dx_t = -\gamma x_t dt + \sigma dW_t
  \]$

- **Use Case**: Simulating volatility bursts, modeling market microstructure.

## 📊 6.6 Stochastic Process Models

### 📈 Geometric Brownian Motion (GBM)

- **Definition**: Classic model for asset prices with constant drift and volatility.
- **Equation**:  
  $\[
  dS_t = \mu S_t dt + \sigma S_t dW_t
  \]$

- **Limitation**: No volatility clustering or mean-reversion.

### 🔄 Heston Model

- **Definition**: Extends GBM with stochastic volatility.
- **Equations**:
  - Asset:  
    $\[
    dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^1
    \]$
  - Variance:  
    $\[
    dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_t^2
    \]$

- **Feature**: Captures volatility smiles and clustering.

### 🔁 Ornstein-Uhlenbeck (OU)

- **Definition**: Mean-reverting process for stationary series.
- **Equation**:  
  $\[
  dx_t = \theta (\mu - x_t) dt + \sigma dW_t
  \]$

- **Use Case**: Interest rates, spreads, log-prices.


### 🔀 Regime-Switching Models

- **Definition**: Time series with discrete volatility regimes.
- **Mechanism**: Hidden Markov model governs transitions.
- **Use Case**: Crisis modeling, volatility regime detection.

---

# 7. 📚 References
1. E. Parzen: "__Stochastic Processes__", 3rd Ed. Dover Publications (2015); S. Aloorravi: "__Metaprogramming with Python__", 1st Ed. Packt (2022); B. Klein, P. Klein: "__Funktionale Programmierung mit Python__", Hanser (2025);
K. Webel, D. Wied: "__Stochastische Prozesse__", 2. Auflage Springer (2016); L. Held: "__Methoden der statistischen Inferenz__", 1. Auflage Spektrum (2008); E. Cinlar: "Stochastic Processes", Dover (2013);
N. Bäuerle, U. Rieder: "__Finanzmathematik in diskreter Zeit__", Springer-Spektrum (2017); M. Albrecht, R. Maurer: "__Investment- und Risikomanagement__", 3. Auflage, Schäffer Poeschel (2008);
N. H. Bingham, R. Kiesel: ""__Risk Neutral Valuation: Pricing and Hedging of Financial Derivatives__", 2. Auflage Springer (2004); T. Björk: ""__Arbitrage Theory in Continuous Time__", 3rd Ed. Oxford University Press (2009);
N. J. Cutland, A. Roux: "__Derivative Pricing in Discrete Time__", Springer (2013); F. Delbaen, W. Schachermayer: "__The Mathematics of Arbitrage__", Springer (2006); 
R. J. Elliott, P. E. Kopp: ""__Mathematics of Financial Markets__", 2nd Ed. Springer (2005); H. Föllmer, A. Scheid: "__A Stochastic Finance: An Introduction in Discrete Time__", 3rd Ed. de Gruyter (2011);
J. C. Hull: "__Options, Futures and Other Derivatives__", 8th Ed. Pearson (2011); J. Kremer: "__Einführung in die diskrete Finanzmathematik__", Springer (2005); 
D. Lamberton, B. Lapeyre: "__Introduction to Stochastic Calculus Applied to Finance__", Chapman & Hall (2007); D. G. Luenberger: "__Investment Science__", Oxford University Press (1998);
S. R. Pliska: "__Introduction to Mathematical Finance: Discrete Time Models__", Blackwell (2000); A. N. Shiryaev: "__Essentials of Stochastic Finance__", World Scientific (2001);
S. E. Shreve: "__Stochastic Calculus for Finance I: The Binomial Asset Pricing Model__", Springer (2004); J. Kremer: "__Portfoliotheorie, Risikomanagement und die Bewertung von Derivaten__", Springer (2011);
L. Rüschendorf: "__Mathematical Risk Analysis__", Springer (2013).
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/8cd6b8f5c7010063936e679806d0ac37cda903c5/AnalyticsEngineerExercise/AnalyticsEngineerTask.ipynb)
3. [![Meta_Multi_Asset_Analysis Report | English](https://img.shields.io/badge/Analytics_Engineering_Gaming_Analysis%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/7b91e2a516ed3974a15aae5d083e0332947e65e6/AnalyticsEngineerExercise/analytics_dashboard.pdf) 
4. A. Meister , T. Sonar: "__Numerik__", 1st Ed. Springer-Spektrum (2019); S. Chapra, R. Canale: "__Numerical Methods for Engineers__", Mcgraw-Hill, 6th Edition (2010). 
5. J. Kilty, A. M. McAllister: "__Mathematical Modeling and Applied Calculus__", 1st Ed. Oxford University Press (2018).
6. U. Kockelkorn: "__Statistik für Anwender__", 1st Ed. Springer (2012), s. chapters 7 - 8.
7. Robert H. Shumway, David S. Stoffer: "__Time Series Analysis and Its Applications with R Examples__", Springer (2011).
8. Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor: "__An Introduction to Statistical Learning with Applications in Python__", Springer (2023).
9. Cornelis W. Oosterlee, Lech A. Grzelak: "__Mathematical Modeling and Computation in Finance with Exercises and Python and MATLAB Computer Codes__", World Scientific (2020).
10. Richard Szeliski: "__Computer Vision - Algorithms and Applications__", Springer (2022).
11. Anthony Scopatz, Kathryn D. Huff: "__Effective Computation in Physics - Field Guide to Research with Python__", O'Reilly Media (2015).
12. Alex Gezerlis: "__Numerical Methods in Physics with Python__", Cambridge University Press (2020).
13. Gary Hutson, Matt Jackson: "__Graph Data Modeling in Python. A practical guide__", Packt-Publishing (2023).
14. Hagen Kleinert: "__Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets__", 5th Edition, World Scientific Publishing Company (2009).
15. Peter Richmond, Jurgen Mimkes, Stefan Hutzler: "__Econophysics and Physical Economics__", Oxford University Press (2013).
16. A. Coryn , L. Bailer Jones: "__Practical Bayesian Inference A Primer for Physical Scientists__", Cambridge University Press (2017).
17. Avram Sidi: "__Practical Extrapolation Methods - Theory and Applications__", Cambridge university Press (2003).
18. Volker Ziemann: "__Physics and Finance__", Springer (2021).
19. Zhi-Hua Zhou: "__Ensemble methods, foundations and algorithms__", CRC Press (2012).
20. B. S. Everitt, et al.: "__Cluster analysis__", Wiley (2011).
21. Lior Rokach, Oded Maimon: "__Data Mining With Decision Trees - Theory and Applications__", World Scientific (2015).
22. Bernhard Schölkopf, Alexander J. Smola: "__Learning with kernels - support vector machines, regularization, optimization and beyond__", MIT Press (2009).
23. Johan A. K. Suykens: "__Regularization, Optimization, Kernels, and Support Vector Machines__", CRC Press (2014).
24. Sarah Depaoli: "__Bayesian Structural Equation Modeling__", Guilford Press (2021).
25. Rex B. Kline: "__Principles and Practice of Structural Equation Modeling__", Guilford Press (2023).
26. Ekaterina Kochmar: "__Getting Started with Natural Language Processing__", Manning (2022).
27. Jakub Langr, Vladimir Bok: "__GANs in Action__", Computer Vision Lead at Founders Factory (2019).
28. David Foster: "__Generative Deep Learning__", O'Reilly(2023).
29. Rowel Atienza: "__Advanced Deep Learning with Keras: Applying GANs and other new deep learning algorithms to the real world__", Packt Publishing (2018).
30. Josh Kalin: "__Generative Adversarial Networks Cookbook__", Packt Publishing (2018).  
31. Thomas Haslwanter: "__Hands-on Signal Analysis with Python: An Introduction__", Springer (2021).
32. Jose Unpingco: "__Python for Signal Processing__", Springer (2023).
33. R. K. Burdick, C. M. Borror, D. C. Montgomery: "__Design and Analysis of Gauge R&R Studies__", 1st Ed. SIAM (2005); 
S. H. Derakhshan , C. V. Deutsch: "__Numerical Integration of Bivariate Gaussian Distribution__", Paper 405, CCG Anual Report 13 (2011).
34. C. Paar, J. Pelzl: "__Understanding Cryptography__", Springer (2010); H. Delfs, H. Knebl: "__Introduction to Cryptography__", 3rd Ed. Springer (2015); J. Katz, Y. lindell: "__Introduction to Modern Cryptography__", 2nd Ed, CRC Press (2015); 
O. Goldreich: "__Foundations of Cryptography__", Cambridge University Press (2008); J. P. Aumasson: "__Serious Cryptography__", no starch press (2018).  
35. J. Berk, P. DeMarzo: „__Corporate Finance__“, 6th Ed., Pearson (2023); R. W. Melicher, E. A. Norton: "__Introduction to Finance__", 16th Ed. WILEY (2017); 
Anatoly B. Schmidt: "__Quantitative Finance for Physicists: An Introduction__", 1st Ed. Academic Press (2005); Alex Backwell: "__An Intuitive Introduction to Finance and Derivatives: Concepts, Terminology and Models__",
 1st Ed, Springer (2023); Michael Isichenko: "__Quantitative Portfolio Management: The Art and Science of Statistical Arbitrage__", 1st Ed., Springer (2021); John H. Cochrane: "__Asset Pricing__", Revised Ed., Princeton University Press (2005);
 Antti Ilmanen: "__Expected Returns: An Investor’s Guide to Harvesting Market Rewards__", 1st Ed., WILEY (2011); Steven E. Shreve: "__Stochastic Calculus for Finance I & II__", 1st Ed., Springer (2004); 
 Andrew Pole: "__Statistical Arbitrage: Algorithmic Trading Insights and Techniques__", 1st Ed., WILEY (2007); Mark S. Joshi: "__The Concepts and Practice of Mathematical Finance__", 2nd Ed., Cambridge University Press (2008);
Kaggle-link: competition-documentation: https://www.kaggle.com/competitions/drw-crypto-market-prediction.
36. R. Nystrom: "__Game Programming Patterns__", 1st Ed. genever benning (2014); A. A. Stepanov, D. E. Rose: "__From Mathematics to Generic Programming__", 1st Ed. Addison-Wesley (2015); 
























