"""
lk_bayes.py
===============

"""

# ================================================================
#  LK-Bayes-Heuristik, Numba-Optimierung, Hamilton-Pfad
# ================================================================

import numpy as np
from typing import Dict, Any, Tuple, List
import numba


# ================================================================
#  Score-Funktion für Kanten
# ================================================================

def edge_score(
    u: int,
    v: int,
    p_post: np.ndarray,
    w_var_post: np.ndarray,
    kl: np.ndarray,
    λ1: float = 2.0,
    λ2: float = 1.0,
    λ3: float = 0.5
) -> float:
    """
    Berechnet den Posterior-Score einer Kante (u, v).

    Score = λ1 * p_post - λ2 * Var - λ3 * KL
    """
    return (
        λ1 * p_post[u, v]
        - λ2 * w_var_post[u, v]
        - λ3 * kl[u, v]
    )


# ================================================================
#  2-opt Move (NumPy)
# ================================================================

def two_opt(cycle: np.ndarray, i: int, k: int) -> np.ndarray:
    """
    Führt einen 2-opt-Move durch, indem der Abschnitt [i:k] invertiert wird.
    """
    new_cycle = cycle.copy()
    new_cycle[i:k] = cycle[i:k][::-1]
    return new_cycle


# ================================================================
#  LK-Bayes-Heuristik (modular, ohne globale Variablen)
# ================================================================

def lk_bayes_cycle(
    p_post: np.ndarray,
    w_var_post: np.ndarray,
    kl: np.ndarray,
    max_iter: int = 200
) -> np.ndarray:
    """
    Führt eine vereinfachte Lin-Kernighan-Heuristik durch,
    die Posterior-Informationen nutzt.

    Parameters
    ----------
    p_post : np.ndarray
        Posterior-Existenzwahrscheinlichkeiten.

    w_var_post : np.ndarray
        Posterior-Gewichtsvarianzen.

    kl : np.ndarray
        KL-Divergenzen.

    max_iter : int
        Maximale Anzahl Iterationen.

    Returns
    -------
    np.ndarray
        Optimierter Hamilton-Zyklus.
    """
    n = p_post.shape[0]
    rng = np.random.default_rng()

    # Startzyklus: zufällige Permutation
    cycle = rng.permutation(n)

    improved = True
    it = 0

    while improved and it < max_iter:
        improved = False
        it += 1

        for i in range(n - 2):
            for k in range(i + 2, n):
                new_cycle = two_opt(cycle, i, k)

                # Score berechnen
                old_score = 0.0
                new_score = 0.0

                for a in range(n - 1):
                    u1, v1 = cycle[a], cycle[a+1]
                    u2, v2 = new_cycle[a], new_cycle[a+1]
                    old_score += edge_score(u1, v1, p_post, w_var_post, kl)
                    new_score += edge_score(u2, v2, p_post, w_var_post, kl)

                if new_score > old_score:
                    cycle = new_cycle
                    improved = True

    return cycle


# ================================================================
#  Zyklus-Stabilitätsanalyse (Numba)
# ================================================================

@numba.njit
def cycle_stability_numba(
    cycle: np.ndarray,
    A: np.ndarray
) -> float:
    """
    Berechnet die Match-Rate eines Hamilton-Zyklus
    gegenüber dem Ist-Graphen A.
    """
    n = cycle.shape[0]
    matches = 0
    for i in range(n - 1):
        if A[cycle[i], cycle[i+1]] == 1:
            matches += 1
    return matches / n


def cycle_stability_full(
    cycle: np.ndarray,
    A: np.ndarray,
    p_post: np.ndarray,
    w_var_post: np.ndarray,
    kl: np.ndarray
) -> Dict[str, float]:
    """
    Berechnet mehrere Stabilitätsmetriken eines Hamilton-Zyklus.

    Returns
    -------
    dict
        match : float
        score : float
        var : float
        kl : float
    """
    n = len(cycle)
    match = cycle_stability_numba(cycle, A)
    score = 0.0
    var_sum = 0.0
    kl_sum = 0.0

    for i in range(n - 1):
        u = cycle[i]
        v = cycle[i+1]
        score += edge_score(u, v, p_post, w_var_post, kl)
        var_sum += w_var_post[u, v]
        kl_sum += kl[u, v]

    return {
        "match": match,
        "score": score,
        "var": var_sum / n,
        "kl": kl_sum / n
    }


# ================================================================
#  Extraktion des stabilsten Hamilton-Pfads
# ================================================================

def extract_stable_hamilton_path(
    cycle: np.ndarray,
    p_post: np.ndarray,
    w_var_post: np.ndarray,
    kl: np.ndarray
) -> List[int]:
    """
    Extrahiert den stabilsten Hamilton-Pfad aus dem finalen Zyklus.

    Der Pfad wird durch die Kanten mit maximalem Posterior-Score bestimmt.
    """
    n = len(cycle)
    scores = []

    for i in range(n - 1):
        u = cycle[i]
        v = cycle[i+1]
        s = edge_score(u, v, p_post, w_var_post, kl)
        scores.append((s, u, v))

    # Sortiere nach Score
    scores.sort(reverse=True)

    # Extrahiere die besten Kanten
    best_edges = scores[: max(3, n // 10)]  # Top 10% oder mindestens 3

    # Baue Pfad
    path = [best_edges[0][1], best_edges[0][2]]
    for _, u, v in best_edges[1:]:
        if path[-1] == u:
            path.append(v)

    return path


# ================================================================
#  Optionaler Cython-Hook (für spätere Optimierung)
# ================================================================

def cython_lk_bayes_hook():
    """
    Platzhalter für spätere Cython-Optimierung.
    """
    pass
