import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial import Polynomial

# -----------------------------
# 🔁 Forecasting Pipeline
# -----------------------------


def run_forecasting_pipeline(csv_path, target_asset,
                             forecast_horizon=50,
                             output_dir="forecast_results"):
    os.makedirs(output_dir, exist_ok=True)

    # Load and filter the asset
    df = pd.read_csv(csv_path)
    if target_asset not in df.columns:
        raise ValueError(f"Asset '{target_asset}' not found in CSV.")

    series = df[target_asset].values

    # -----------------------------
    # 🔁 1. Kalman Filtering
    # -----------------------------
    def kalman_filter(series):
        n = len(series)
        x = np.zeros(n)
        P = np.zeros(n)
        Q = 1e-5
        R = np.var(series[:100])
        x[0] = series[0]
        P[0] = 1.0
        for t in range(1, n):
            x_pred = x[t-1]
            P_pred = P[t-1] + Q
            K = P_pred / (P_pred + R)
            x[t] = x_pred + K * (series[t] - x_pred)
            P[t] = (1 - K) * P_pred
        return x, P

    filtered, cov = kalman_filter(series)

    # -----------------------------
    # 📈 2. Richardson Extrapolation
    # -----------------------------
    def richardson_extrapolation(x, h=1):
        base = x[-forecast_horizon:]
        extrapolated = []
        for i in range(forecast_horizon):
            if i + 2*h < len(base):
                A_h = base[i]
                A_h2 = base[i + h]
                A_extrap = A_h2 + (A_h2 - A_h) / (2**1 - 1)
                extrapolated.append(A_extrap)
            else:
                extrapolated.append(base[-1])
        return np.array(extrapolated)

    extrapolated = richardson_extrapolation(filtered)

    # -----------------------------
    # 🌀 3. Anti-Limit Stabilization
    # -----------------------------
    stabilized = gaussian_filter1d(extrapolated, sigma=2)

    # -----------------------------
    # ⚛️ 4. Quantum Noise Mitigation (Simulated)
    # -----------------------------
    def zero_noise_extrapolation(base_forecast, noise_levels=[0.1, 0.2, 0.3]):
        noisy_versions = []
        for noise in noise_levels:
            noise_array = np.random.normal(0, noise, size=len(base_forecast))
            noisy_versions.append(base_forecast + noise_array)

        denoised = []
        for i in range(len(base_forecast)):
            y = [noisy[i] for noisy in noisy_versions]
            p = Polynomial.fit(noise_levels, y, deg=2)
            denoised.append(p(0))
        return np.array(denoised), noisy_versions

    denoised, noisy_versions = zero_noise_extrapolation(stabilized)

    # -----------------------------
    # 🔄 5. Hybrid Switching
    # -----------------------------
    def hybrid_switch(filtered, denoised, cov, threshold=0.05):
        hybrid = []
        for i in range(len(denoised)):
            if np.mean(cov[-forecast_horizon:]) > threshold:
                hybrid.append(denoised[i])
            else:
                hybrid.append(filtered[-forecast_horizon + i])
        return np.array(hybrid)

    hybrid_forecast = hybrid_switch(filtered, denoised, cov)

    # -----------------------------
    # 📊 Plotting
    # -----------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(series, label="Raw")
    axs[0, 0].plot(filtered, label="Kalman Filtered")
    axs[0, 0].set_title(f"{target_asset}: Raw vs Filtered")
    axs[0, 0].legend()

    axs[0, 1].plot(extrapolated, label="Richardson")
    axs[0, 1].plot(stabilized, label="Stabilized")
    axs[0, 1].set_title(f"{target_asset}: Extrapolated vs Stabilized")
    axs[0, 1].legend()

    for noisy in noisy_versions:
        axs[1, 0].plot(noisy, alpha=0.3, color='gray')
    axs[1, 0].plot(denoised, label="Denoised", color='blue')
    axs[1, 0].set_title(f"{target_asset}: Noisy Ensemble vs Denoised")
    axs[1, 0].legend()

    axs[1, 1].plot(hybrid_forecast, label="Hybrid Forecast", color='green')
    axs[1, 1].set_title(f"{target_asset}: Hybrid Forecast")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{target_asset}_forecast.png")
    plt.close()

    # -----------------------------
    # 📁 Export CSV
    # -----------------------------
    forecast_df = pd.DataFrame({
        "Richardson": extrapolated,
        "Stabilized": stabilized,
        "Denoised": denoised,
        "Hybrid": hybrid_forecast
    })
    forecast_df.to_csv(f"{output_dir}/{target_asset}_forecast.csv",
                       index=False)

    print(f"✅ Forecast for {target_asset} saved to '{output_dir}'.")
