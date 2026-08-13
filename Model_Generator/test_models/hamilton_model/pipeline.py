"""
pipeline.py
====================

Dieses Modul kapselt die Streaming-Pipeline für Hamilton-Zyklen-Analysen.
Es orchestriert:

    - den Datengenerator (aus generators.py)
    - die Posterior-Engine (aus posterior.py)
    - die LK-Bayes-Heuristik (aus lk_bayes.py)
    - die Drift-Detektion
    - die CSV-Sampling-Logik
    - die Zeitreihen-Speicherung
    - optionale Visualisierung (Plot-Funktionen aus graphviz_plot.py)

Dieses Modul enthält KEINE Modelllogik, KEINE Generatoren und KEINE
Hamilton-Zyklen-Heuristik. Es ist rein für die Pipeline zuständig.

Die Hauptfunktion ist:

    run_pipeline(...)

Diese wird später von model.py aufgerufen.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, List, Tuple

# Importiere Modellkomponenten
from .posterior import (
    update_posterior,
    compute_drift,
    posterior_stats,
    build_posterior_graph,
    p_post,
    w_mean_post,
    w_var_post,
    w_mean_prev,
    w_var_prev
)

from .lk_bayes import (
    lk_bayes_cycle,
    cycle_stability_full,
    extract_stable_hamilton_path
)

from .graphviz_plot import plot_hamilton_graphviz


# ================================================================
#  CSV Logger (Pipeline-spezifisch)
# ================================================================

def init_csv_logger() -> List[Dict[str, Any]]:
    """
    Initialisiert die CSV-Sample-Liste.

    Returns
    -------
    list[dict]
        Leere Liste für CSV-Samples.
    """
    return []


def log_csv_sample(
    csv_rows: List[Dict[str, Any]],
    t: int,
    M: Dict[str, Any],
    posterior_stats: Dict[str, float],
    cycle_stats: Dict[str, float],
    drift_score: float
) -> None:
    """
    Speichert einen CSV-Sample-Eintrag.

    Parameters
    ----------
    csv_rows : list[dict]
        Liste der CSV-Samples.

    t : int
        Zeitschritt.

    M : dict
        Metadaten des Streams.

    posterior_stats : dict
        Posterior-Statistiken.

    cycle_stats : dict
        Hamilton-Zyklus-Statistiken.

    drift_score : float
        Drift-Score.
    """
    row = {
        "t": t,
        "phase": M["phase"],
        "drift": M["drift"],
        "drift_type": M["drift_type"],
        "mean_p_true": float(np.mean(M["p_true"])),
        "mean_w_true": float(np.mean(M["F_true"][0])),
        "mean_p_post": posterior_stats["mean_p"],
        "mean_w_post": posterior_stats["mean_w"],
        "H_score": cycle_stats["match"],
        "drift_score": drift_score
    }
    csv_rows.append(row)


# ================================================================
#  Zeitreihen-Container
# ================================================================

def init_timeseries() -> Dict[str, List[float]]:
    """
    Initialisiert alle Zeitreihen für die Pipeline.

    Returns
    -------
    dict[str, list]
        Dictionary mit leeren Zeitreihen.
    """
    return {
        "t": [],
        "mean_p": [],
        "mean_w": [],
        "H_score": [],
        "drift": [],
        "cycle_match": [],
        "cycle_score": [],
        "cycle_var": [],
        "cycle_kl": []
    }


def append_timeseries(
    ts: Dict[str, List[float]],
    t: int,
    pstats: Dict[str, float],
    cstats: Dict[str, float],
    drift_score: float
) -> None:
    """
    Fügt einen Zeitschritt zu den Zeitreihen hinzu.

    Parameters
    ----------
    ts : dict[str, list]
        Zeitreihen.

    t : int
        Zeitschritt.

    pstats : dict
        Posterior-Statistiken.

    cstats : dict
        Zyklus-Statistiken.

    drift_score : float
        Drift-Score.
    """
    ts["t"].append(t)
    ts["mean_p"].append(pstats["mean_p"])
    ts["mean_w"].append(pstats["mean_w"])
    ts["H_score"].append(cstats["match"])
    ts["drift"].append(drift_score)
    ts["cycle_match"].append(cstats["match"])
    ts["cycle_score"].append(cstats["score"])
    ts["cycle_var"].append(cstats["var"])
    ts["cycle_kl"].append(cstats["kl"])


# ================================================================
#  Hauptpipeline
# ================================================================

def run_pipeline(
    generator: Callable[..., Any],
    n: int,
    T: int,
    sample_rate: int = 10,
    threshold: float = 0.5,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[int], Dict[str, np.ndarray]]:
    """
    Führt die Streaming-Pipeline aus.

    Parameters
    ----------
    generator : callable
        Datengenerator-Funktion.

    n : int
        Anzahl der Knoten.

    T : int
        Anzahl der Zeitschritte.

    sample_rate : int
        Jeder sample_rate-te Schritt wird als CSV-Sample gespeichert.

    threshold : float
        Posterior-Schwelle für Kantenexistenz.

    verbose : bool
        Fortschrittsausgabe.

    Returns
    -------
    df : pd.DataFrame
        Zeitreihen.

    df_csv : pd.DataFrame
        CSV-Samples.

    cycle : np.ndarray
        Finaler Hamilton-Zyklus.

    stable_path : list[int]
        Stabilster Hamilton-Pfad.

    posterior : dict[str, np.ndarray]
        Posterior-Parameter.
    """

    # ------------------------------------------------------------
    # 1. Generator starten
    # ------------------------------------------------------------
    stream = generator(n=n, T=T)

    # ------------------------------------------------------------
    # 2. CSV-Logger & Zeitreihen initialisieren
    # ------------------------------------------------------------
    csv_rows = init_csv_logger()
    ts = init_timeseries()

    # ------------------------------------------------------------
    # 3. Streaming-Schleife
    # ------------------------------------------------------------
    for sample in stream:
        A = sample["A"]
        W = sample["W"]
        M = sample["M"]
        t = M["t"]

        if verbose and t % 50 == 0:
            print(f"[Pipeline] t = {t}/{T}")

        # Posterior-Update
        update_posterior(A, W)

        # Drift
        drift_score = compute_drift()

        # Posterior-Graph
        B_post, W_post = build_posterior_graph(p_post, w_mean_post, threshold)

        # KL-Matrix
        kl = (w_mean_post - w_mean_prev) ** 2 + (w_var_post - w_var_prev) ** 2

        # Hamilton-Zyklus
        cycle = lk_bayes_cycle(p_post, w_var_post, kl)

        # Zyklus-Stabilität
        cstats = cycle_stability_full(cycle, A, p_post, w_var_post, kl)

        # Posterior-Statistiken
        pstats = posterior_stats()

        # CSV-Sampling
        if t % sample_rate == 0:
            log_csv_sample(csv_rows, t, M, pstats, cstats, drift_score)

        # Zeitreihen speichern
        append_timeseries(ts, t, pstats, cstats, drift_score)

    # ------------------------------------------------------------
    # 4. Stabilsten Hamilton-Pfad extrahieren
    # ------------------------------------------------------------
    stable_path = extract_stable_hamilton_path(cycle, p_post, w_var_post, kl)

    # ------------------------------------------------------------
    # 5. DataFrames erzeugen
    # ------------------------------------------------------------
    df = pd.DataFrame(ts)
    df_csv = pd.DataFrame(csv_rows)

    # ------------------------------------------------------------
    # 6. Posterior zurückgeben
    # ------------------------------------------------------------
    posterior = {
        "p_post": p_post,
        "w_mean_post": w_mean_post,
        "w_var_post": w_var_post
    }

    return df, df_csv, cycle, stable_path, posterior
