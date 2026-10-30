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
