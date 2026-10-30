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
