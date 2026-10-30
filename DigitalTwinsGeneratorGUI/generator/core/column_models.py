# generator/core/column_models.py

import numpy as np
import random
from datetime import datetime, timedelta


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def _smooth_noise(n, scale=1.0):
    """Generates smooth noise using cumulative sum of Gaussian noise."""
    return np.cumsum(np.random.normal(0, scale, n))


def _bounded(values, low, high):
    """Clips values to a given range."""
    return np.clip(values, low, high)


# ---------------------------------------------------------
# Numeric Sensor Models
# ---------------------------------------------------------

def simulate_temperature(n, col, config):
    """
    Temperature fluctuates slowly with smooth noise and slight drift.
    """
    base = 40 + _smooth_noise(n, scale=0.05)
    return _bounded(base, 20, 90)


def simulate_rpm(n, col, config):
    """
    Motor RPM oscillates around a nominal value with periodic variation.
    """
    t = np.linspace(0, 4 * np.pi, n)
    base = 1500 + 200 * np.sin(t) + np.random.normal(0, 20, n)
    return _bounded(base, 0, 6000).astype(int)


def simulate_vibration(n, col, config):
    """
    Vibration level: low baseline with occasional spikes.
    """
    base = np.abs(np.random.normal(0.2, 0.05, n))
    spikes = np.random.choice([0, 1], size=n, p=[0.98, 0.02]) * np.random.uniform(1, 3, n)
    return base + spikes


def simulate_power(n, col, config):
    """
    Power consumption correlates with RPM + noise.
    """
    rpm = simulate_rpm(n, col, config)
    power = 0.02 * rpm + np.random.normal(0, 5, n)
    return _bounded(power, 0, 5000)


def simulate_voltage(n, col, config):
    """
    Voltage: stable with tiny noise.
    """
    return 230 + np.random.normal(0, 0.5, n)


def simulate_current(n, col, config):
    """
    Current: correlated with power consumption.
    """
    power = simulate_power(n, col, config)
    current = power / 230 + np.random.normal(0, 0.1, n)
    return _bounded(current, 0, 50)


def simulate_pressure(n, col, config):
    """
    Pressure/load: slow drift + noise.
    """
    base = 50 + _smooth_noise(n, scale=0.1)
    return _bounded(base, 0, 200)


def simulate_noise(n, col, config):
    """
    Noise level: random with occasional peaks.
    """
    base = np.random.normal(40, 2, n)
    peaks = np.random.choice([0, 1], size=n, p=[0.97, 0.03]) * np.random.uniform(10, 20, n)
    return base + peaks


# ---------------------------------------------------------
# Categorical / Boolean Models
# ---------------------------------------------------------

def simulate_onoff(n, col, config):
    """Boolean on/off with 90% uptime."""
    return np.random.choice([0, 1], size=n, p=[0.1, 0.9])


def simulate_mode(n, col, config):
    """Operating mode: Idle, Low, High."""
    categories = col.get("categories", ["Idle", "Low", "High"])
    probs = [0.2, 0.5, 0.3]
    return np.random.choice(categories, size=n, p=probs)


def simulate_error(n, col, config):
    """Error code: None, Minor, Major."""
    categories = col.get("categories", ["None", "Minor", "Major"])
    probs = [0.95, 0.04, 0.01]
    return np.random.choice(categories, size=n, p=probs)


def simulate_interlock(n, col, config):
    """Safety interlock: mostly off."""
    return np.random.choice([0, 1], size=n, p=[0.98, 0.02])


# ---------------------------------------------------------
# Auxiliary Models
# ---------------------------------------------------------

def simulate_timestamp(n, col, config):
    """
    Generates ISO8601 timestamps at the sampling frequency.
    """
    freq_hz = config.get("frequency_hz", 10)
    dt = 1.0 / freq_hz

    start = datetime.utcnow()
    return [
        (start + timedelta(seconds=i * dt)).isoformat()
        for i in range(n)
    ]


def simulate_log(n, col, config):
    """
    Generates simple log messages.
    """
    messages = [
        "System OK",
        "Temperature stable",
        "RPM nominal",
        "Minor fluctuation detected",
        "Sensor check passed",
        "No anomalies detected"
    ]
    return np.random.choice(messages, size=n)


def simulate_cycle(n, col, config):
    """
    Monotonically increasing cycle counter.
    """
    return np.arange(n)


# ---------------------------------------------------------
# Lookup Table
# ---------------------------------------------------------

COLUMN_MODEL_MAP = {
    "simulate_temperature": simulate_temperature,
    "simulate_rpm": simulate_rpm,
    "simulate_vibration": simulate_vibration,
    "simulate_power": simulate_power,
    "simulate_voltage": simulate_voltage,
    "simulate_current": simulate_current,
    "simulate_pressure": simulate_pressure,
    "simulate_noise": simulate_noise,

    "simulate_onoff": simulate_onoff,
    "simulate_mode": simulate_mode,
    "simulate_error": simulate_error,
    "simulate_interlock": simulate_interlock,

    "simulate_timestamp": simulate_timestamp,
    "simulate_log": simulate_log,
    "simulate_cycle": simulate_cycle,
}