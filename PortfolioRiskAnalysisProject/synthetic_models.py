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
