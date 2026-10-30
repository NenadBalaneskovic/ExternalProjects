#  import sys
import os
import numpy as np
import pandas as pd
import pyqtgraph as pg
import matplotlib.pyplot as plt
import yaml
import importlib.util

from PyQt5.QtWidgets import (
    # QApplication,
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QListWidget, QListWidgetItem,
    QFileDialog, QCheckBox, QFrame, QStatusBar, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal  # , QAbstractTableModel
from functools import partial
from scipy.stats import norm
from toolz import pipe

# ---------------- Registries ---------------- #
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

# ---------------- Plugin Discovery ---------------- #


def discover_plugins(plugin_dir="plugins"):
    if not os.path.exists(plugin_dir):
        return
    for filename in os.listdir(plugin_dir):
        if filename.endswith(".py"):
            path = os.path.join(plugin_dir, filename)
            spec = importlib.util.spec_from_file_location(filename[:-3], path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

# ---------------- Schema Validation ---------------- #


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

# ---------------- Synthetic Models ---------------- #


def simulate_gbm(n_assets=5, n_steps=1000, mu=0.1, sigma=0.2, corr=0.5):
    dt = 1/252
    cov_matrix = np.full((n_assets, n_assets), corr)
    np.fill_diagonal(cov_matrix, 1.0)
    L = np.linalg.cholesky(cov_matrix)
    returns = np.random.normal(0, 1, (n_steps, n_assets)) @ L.T
    prices = np.full((n_steps, n_assets), 100.0)
    for t in range(1, n_steps):
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt +
                                         sigma * np.sqrt(dt) * returns[t])
    return pd.DataFrame(prices, columns=[f"Asset_{i+1}" for i in
                                         range(n_assets)])


def simulate_ou(n_assets=5, n_steps=1000, theta=0.15, mu=0.0, sigma=0.3):
    dt = 1/252
    x = np.zeros((n_steps, n_assets))
    for t in range(1, n_steps):
        x[t] = (x[t-1] + theta * (mu - x[t-1]) * dt +
                sigma * np.sqrt(dt) * np.random.normal(size=n_assets))
    return pd.DataFrame(x, columns=[f"OU_{i+1}" for i in range(n_assets)])


def simulate_heston(n_assets=1, n_steps=1000, mu=0.05, kappa=0.5, theta=0.04,
                    xi=0.1, rho=-0.7):
    dt = 1/252
    S = np.full((n_steps, n_assets), 100.0)
    v = np.full((n_steps, n_assets), theta)
    for t in range(1, n_steps):
        z1 = np.random.normal(size=n_assets)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(size=n_assets)
        v[t] = (np.abs(v[t-1] + kappa * (theta - v[t-1]) * dt +
                       xi * np.sqrt(v[t-1] * dt) * z2))
        S[t] = (S[t-1] * np.exp((mu - 0.5 * v[t-1]) * dt +
                                np.sqrt(v[t-1] * dt) * z1))
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

# ---------------- Generator Dispatcher ---------------- #


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

# ---------------- YAML Schema I/O ---------------- #


def load_yaml_schema(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml_schema(schema, path):
    with open(path, "w") as f:
        yaml.dump(schema, f)

# ---------------- Risk Models ---------------- #


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
    sim_returns = np.random.normal(df.mean(), df.std(),
                                   (n_sim, len(df.columns)))
    return pd.Series(np.percentile(sim_returns,
                                   (1 - confidence) * 100, axis=0),
                     index=df.columns)


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

# ---------------- Risk Pipeline ---------------- #


def risk_pipeline(df, model_name, **kwargs):
    model = RISK_MODEL_REGISTRY.get(model_name)
    if model:
        return pipe(df, partial(model, **kwargs))
    return None

# ---------------- Physics Models ---------------- #


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

# ---------------- Physics Pipeline ---------------- #


def physics_pipeline(df, model_name, **kwargs):
    model = PHYSICS_MODEL_REGISTRY.get(model_name)
    if model:
        return pipe(df, partial(model, **kwargs))
    return None


class QuantCanvas(QMainWindow):
    theme_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FIN QuantLab — QuantCanvas")
        self.resize(1400, 900)
        self.df = None
        self.schema = {}
        self.simulation_schema = {}

        discover_plugins()

        # Main layout
        central = QWidget()
        self.layout = QVBoxLayout()
        central.setLayout(self.layout)
        self.setCentralWidget(central)

        # Top Controls
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

        # Asset Selector
        self.asset_list = QListWidget()
        self.asset_list.setDragEnabled(True)
        self.asset_list.setSelectionMode(QListWidget.MultiSelection)
        self.asset_list.setFixedHeight(100)
        self.layout.addWidget(QLabel("Assets"))
        self.layout.addWidget(self.asset_list)

        # Dynamic Widget Panel
        self.widget_panel = QFrame()
        self.widget_layout = QVBoxLayout()
        self.widget_panel.setLayout(self.widget_layout)
        self.layout.addWidget(QLabel("Schema Widgets"))
        self.layout.addWidget(self.widget_panel)

        # Chart Panel
        self.chart = pg.PlotWidget(title="Real-Time Chart")
        self.layout.addWidget(self.chart)

        # Status Bar
        status_bar = QStatusBar()
        status_bar.addWidget(QLabel("Ready"))
        progress = QProgressBar()
        progress.setValue(0)
        status_bar.addPermanentWidget(progress)
        self.setStatusBar(status_bar)

    def toggle_theme(self, state):
        if state == Qt.Checked:
            self.setStyleSheet("background-color: #2b2b2b; color: white;")
        else:
            self.setStyleSheet("")

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "",
                                              "CSV Files (*.csv)")
        if path:
            self.df = pd.read_csv(path, parse_dates=True)
            try:
                validate_schema(self.df)
            except Exception as e:
                print(f"Schema validation failed: {e}")
                return
            self.schema = parse_schema(self.df)
            self.populate_assets()
            self.generate_widgets()
            self.update_chart()

    def import_yaml_schema(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import YAML", "",
                                              "YAML Files (*.yaml *.yml)")
        if path:
            self.simulation_schema = load_yaml_schema(path)
            self.df = generate_synthetic_data(**self.simulation_schema)
            self.schema = parse_schema(self.df)
            self.populate_assets()
            self.generate_widgets()
            self.update_chart()

    def export_yaml_schema(self):
        if not self.simulation_schema:
            print("No schema to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export YAML", "",
                                              "YAML Files (*.yaml *.yml)")
        if path:
            save_yaml_schema(self.simulation_schema, path)
            print(f"Exported YAML schema to {path}")

    def populate_assets(self):
        self.asset_list.clear()
        for col in self.df.columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsDragEnabled)
            self.asset_list.addItem(item)

    def generate_widgets(self):
        for i in reversed(range(self.widget_layout.count())):
            widget = self.widget_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for col, dtype in self.schema.items():
            label = QLabel(f"{col} ({dtype})")
            label.setStyleSheet("font-weight: bold;")
            self.widget_layout.addWidget(label)

            if dtype == "numeric":
                combo = QComboBox()
                combo.addItems(["mean", "std", "min", "max"])
                combo.currentIndexChanged.connect(lambda _, c=col,
                                                  w=combo:
                                                      self.apply_metric(c, w))
                self.widget_layout.addWidget(combo)
            elif dtype == "categorical":
                combo = QComboBox()
                combo.addItems(self.df[col].astype(str).unique().tolist())
                combo.currentIndexChanged.connect(lambda _, c=col,
                                                  w=combo:
                                                      self.filter_category(c,
                                                                           w))
                self.widget_layout.addWidget(combo)
            elif dtype == "datetime":
                self.widget_layout.addWidget(QLabel(
                    "Time filter coming soon..."))

    def apply_metric(self, column, widget):
        metric = widget.currentText()
        if metric and column in self.df.columns:
            value = getattr(self.df[column], metric)()
            print(f"{metric} of {column}: {value}")

    def filter_category(self, column, widget):
        value = widget.currentText()
        if value:
            filtered = self.df[self.df[column].astype(str) == value]
            print(f"Filtered {column} = {value}, {len(filtered)} rows")

    def update_chart(self):
        if self.df is None:
            return
        selected_items = self.asset_list.selectedItems()
        selected_cols = [item.text() for item in selected_items if
                         item.text() in self.df.columns]

        self.chart.clear()
        for col in selected_cols:
            self.chart.plot(self.df[col][:500],
                            pen=pg.mkPen(width=2), name=col)

    def export_dataset(self):
        if self.df is None:
            print("No data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save File", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;PDF Files (*.pdf)")
        if path:
            if path.endswith(".csv"):
                self.df.to_csv(path, index=False)
            elif path.endswith(".xlsx"):
                self.df.to_excel(path, index=False)
            elif path.endswith(".pdf"):
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.axis('off')
                ax.table(cellText=self.df.head(20).values,
                         colLabels=self.df.columns, loc='center')
                fig.savefig(path)
            print(f"Exported to {path}")

# ---------------- Main Entry ---------------- #


if __name__ == "__main__":
    print("Starting GUI...")

    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = QuantCanvas()
    window.show()
    sys.exit(app.exec_())

