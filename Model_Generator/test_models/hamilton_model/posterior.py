""""
posterior.py
===============

""""

# ================================================================
#  Basis-Codeblock für ExtractHamiltonCycles(...)
#  Enthält:
#    - Imports
#    - Posterior-State-Objekt (statt globaler Variablen)
#    - Hilfsfunktionen
#    - Posterior-Engine
#    - Drift-Detektion
#    - CSV-Logger
#    - Plot-Funktionen
#    - Posterior-Graph-Konstruktion
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, Tuple, List


# ================================================================
#  Notebook-Plot-Style
# ================================================================
plt.style.use("seaborn-v0_8")


# ================================================================
#  Posterior-State-Objekt
# ================================================================

class PosteriorState:
    """
    Kapselt alle Posterior-Parameter, die früher global waren.
    Dadurch ist die Pipeline modular, testbar und exportierbar.
    """

    def __init__(self, n: int):
        self.p_post = np.full((n, n), 0.5, dtype=np.float32)
        self.w_mean_post = np.full((n, n), 0.5, dtype=np.float32)
        self.w_var_post = np.full((n, n), 0.05, dtype=np.float32)

        self.w_mean_prev = self.w_mean_post.copy()
        self.w_var_prev = self.w_var_post.copy()

    def update(self, A: np.ndarray, W: np.ndarray) -> None:
        """
        Aktualisiert Posterior-Parameter.
        """
        # p_post Update
        self.p_post = 0.99 * self.p_post + 0.01 * A

        # vorherige Werte speichern
        self.w_mean_prev = self.w_mean_post.copy()
        self.w_var_prev = self.w_var_post.copy()

        # Gewichtsposterior
        self.w_mean_post = 0.99 * self.w_mean_post + 0.01 * W
        self.w_var_post = 0.99 * self.w_var_post + 0.01 * (W - self.w_mean_post) ** 2

    def drift_score(self) -> float:
        """
        KL-basierter Drift-Score.
        """
        diff = (self.w_mean_post - self.w_mean_prev) ** 2 + \
               (self.w_var_post - self.w_var_prev) ** 2
        return float(np.mean(diff))

    def stats(self) -> Dict[str, float]:
        """
        Posterior-Statistiken.
        """
        return {
            "mean_p": float(np.mean(self.p_post)),
            "mean_w": float(np.mean(self.w_mean_post))
        }


# ================================================================
#  CSV Logger (nicht global!)
# ================================================================

def log_csv(
    csv_rows: List[Dict[str, Any]],
    t: int,
    M: Dict[str, Any],
    posterior_stats: Dict[str, float],
    cycle_stats: Dict[str, float],
    drift_score: float
) -> None:
    """
    Speichert einen CSV-Sample-Eintrag.
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
        "H_score": cycle_stats["H_score"],
        "drift_score": drift_score
    }
    csv_rows.append(row)


# ================================================================
#  Plot-Funktionen
# ================================================================

def plot_time_series(df: pd.DataFrame, column: str, title: str) -> None:
    """
    Plottet eine Zeitreihe.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(df["t"], df[column], label=column)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel(column)
    plt.grid(True)
    plt.legend()
    plt.show()


# ================================================================
#  Posterior-Graph-Konstruktion
# ================================================================

def build_posterior_graph(
    p_post: np.ndarray,
    w_mean_post: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erzeugt einen gewichteten Posterior-Graphen.
    """
    B = (p_post > threshold).astype(np.uint8)
    Wp = w_mean_post.copy()
    return B, Wp
