"""
model.py
====================

Dieses Modul kapselt die Hauptfunktion ExtractHamiltonCycles(...),
die die gesamte Pipeline orchestriert.

Sie ruft:
    - generator (aus generators.py)
    - run_pipeline (aus pipeline.py)
    - plot_time_series (aus plots.py)
    - plot_hamilton_graphviz (aus graphviz_plot.py)

"""

from __future__ import annotations
from typing import Callable, Dict, Any
import pandas as pd

from .pipeline import run_pipeline
from .plots import plot_time_series
from .graphviz_plot import plot_hamilton_graphviz


def ExtractHamiltonCycles(
    generator: Callable[..., Any],
    n: int = 300,
    T: int = 500,
    sample_rate: int = 10,
    threshold: float = 0.5,
    plot: bool = True,
    save_csv: bool = True,
    return_results: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Führt die vollständige Hamilton-Zyklus-Analyse-Pipeline aus.
    """

    # ------------------------------------------------------------
    # 1. Pipeline ausführen
    # ------------------------------------------------------------
    df, df_csv, cycle, stable_path, posterior = run_pipeline(
        generator=generator,
        n=n,
        T=T,
        sample_rate=sample_rate,
        threshold=threshold,
        verbose=verbose
    )

    # ------------------------------------------------------------
    # 2. CSV speichern
    # ------------------------------------------------------------
    if save_csv:
        df_csv.to_csv("hamilton_stream_samples.csv", index=False)
        if verbose:
            print("CSV-Samples gespeichert: hamilton_stream_samples.csv")

    # ------------------------------------------------------------
    # 3. Visualisierung
    # ------------------------------------------------------------
    if plot:
        print("Erzeuge Plots...")

        plot_time_series(df, "mean_p", "Posterior Mean p(t)")
        plot_time_series(df, "mean_w", "Posterior Mean Weight(t)")
        plot_time_series(df, "H_score", "Hamilton Cycle Match Rate")
        plot_time_series(df, "drift", "Drift Score")
        plot_time_series(df, "cycle_score", "Hamilton Cycle Posterior Score")
        plot_time_series(df, "cycle_var", "Hamilton Cycle Variance Stability")
        plot_time_series(df, "cycle_kl", "Hamilton Cycle KL Stability")

        # Graphviz-Plot
        dot = plot_hamilton_graphviz(
            cycle=cycle,
            p_post=posterior["p_post"],
            w_mean_post=posterior["w_mean_post"],
            w_var_post=posterior["w_var_post"],
            step=T,
            title="Stabilster Hamilton-Zyklus"
        )
        display(dot)

    # ------------------------------------------------------------
    # 4. Rückgabe
    # ------------------------------------------------------------
    if return_results:
        return {
            "df": df,
            "csv_samples": df_csv,
            "stable_cycle": cycle,
            "stable_path": stable_path,
            "posterior": posterior
        }
