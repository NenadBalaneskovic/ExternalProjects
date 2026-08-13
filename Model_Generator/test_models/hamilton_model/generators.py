"""
generators.py
====================

Dieses Modul enthält drei Generatorfunktionen zur Erzeugung großer
gewichteter Graphstreams für Hamilton-Zyklen-Analysen.

Alle Generatoren erzeugen pro Zeitschritt ein Dictionary:

    {
        "A": A_t,          # np.ndarray (n,n), dtype=uint8
        "W": W_t,          # np.ndarray (n,n), dtype=float32
        "M": M_t           # Metadaten
    }

Die Generatoren sind speichereffizient und skalieren bis in den
Gigabyte-Bereich, da sie die Daten als Stream erzeugen und nicht
persistieren.

Generatoren:
    - generate_stream_stable
    - generate_stream_drift_training
    - generate_stream_drift_prediction
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Generator, Optional


# ================================================================
#  Hilfsfunktionen für Drift
# ================================================================

def drift_additive(x: np.ndarray, step: float) -> np.ndarray:
    """
    Additiver Drift: x -> x + step

    Parameters
    ----------
    x : np.ndarray
        Eingabearray.

    step : float
        Additiver Drift.

    Returns
    -------
    np.ndarray
        Gedriftetes Array.
    """
    return np.clip(x + step, 0.0, 1.0)


def drift_multiplicative(x: np.ndarray, factor: float) -> np.ndarray:
    """
    Multiplikativer Drift: x -> x * factor

    Parameters
    ----------
    x : np.ndarray
        Eingabearray.

    factor : float
        Multiplikationsfaktor.

    Returns
    -------
    np.ndarray
        Gedriftetes Array.
    """
    return np.clip(x * factor, 0.0, 1.0)


def drift_random(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Zufälliger Drift: x -> x + Normalverteilung(0, sigma)

    Parameters
    ----------
    x : np.ndarray
        Eingabearray.

    sigma : float
        Standardabweichung der Drift.

    Returns
    -------
    np.ndarray
        Gedriftetes Array.
    """
    return np.clip(x + np.random.normal(0, sigma, size=x.shape), 0.0, 1.0)


# ================================================================
#  1. Generator: Stabile Quelle (Best Case)
# ================================================================

def generate_stream_stable(
    n: int,
    T: int,
    p_init: Optional[np.ndarray] = None,
    F_init: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Generator[Dict[str, object], None, None]:
    """
    Erzeugt einen stabilen Graphstream ohne Drift.

    Parameters
    ----------
    n : int
        Anzahl der Knoten.

    T : int
        Anzahl der Zeitschritte.

    p_init : np.ndarray, optional
        Initiale Existenzwahrscheinlichkeiten.

    F_init : (w_mean, w_var), optional
        Initiale Gewichtserwartungen und -varianzen.

    Yields
    ------
    dict
        Ein Dictionary mit:
            - "A": Inzidenzmatrix (uint8)
            - "W": Gewichtsmatrix (float32)
            - "M": Metadaten
    """
    rng = np.random.default_rng()

    # Initiale Existenzwahrscheinlichkeiten
    if p_init is None:
        p_true = rng.uniform(0.1, 0.9, size=(n, n)).astype(np.float32)
    else:
        p_true = p_init.astype(np.float32)

    # Initiale Gewichtverteilung
    if F_init is None:
        w_mean_true = rng.uniform(0.2, 0.8, size=(n, n)).astype(np.float32)
        w_var_true = rng.uniform(0.01, 0.05, size=(n, n)).astype(np.float32)
    else:
        w_mean_true, w_var_true = F_init

    for t in range(T):
        # Kantenexistenz
        A_t = (rng.random((n, n)) < p_true).astype(np.uint8)

        # Gewichte
        W_t = w_mean_true + rng.normal(0, np.sqrt(w_var_true), size=(n, n))
        W_t = np.clip(W_t, 0.0, 1.0).astype(np.float32)

        # Metadaten
        M_t = {
            "t": t,
            "phase": "train" if t < T // 2 else "predict",
            "drift": False,
            "drift_type": "none",
            "p_true": p_true,
            "F_true": (w_mean_true, w_var_true),
        }

        yield {"A": A_t, "W": W_t, "M": M_t}


# ================================================================
#  2. Generator: Drift während der Trainingsphase
# ================================================================

def generate_stream_drift_training(
    n: int,
    T: int,
    T_train: int,
    drift_strength: float = 0.01
) -> Generator[Dict[str, object], None, None]:
    """
    Erzeugt einen Graphstream mit Drift während der Trainingsphase.

    Parameters
    ----------
    n : int
        Anzahl der Knoten.

    T : int
        Anzahl der Zeitschritte.

    T_train : int
        Länge der Trainingsphase.

    drift_strength : float
        Stärke des Drifts.

    Yields
    ------
    dict
        Stream-Dictionary (A, W, M).
    """
    rng = np.random.default_rng()

    # Initiale Parameter
    p_true = rng.uniform(0.1, 0.9, size=(n, n)).astype(np.float32)
    w_mean_true = rng.uniform(0.2, 0.8, size=(n, n)).astype(np.float32)
    w_var_true = rng.uniform(0.01, 0.05, size=(n, n)).astype(np.float32)

    for t in range(T):

        # Drift nur während Training
        if t < T_train:
            p_true = drift_random(p_true, drift_strength)
            w_mean_true = drift_random(w_mean_true, drift_strength)
            w_var_true = drift_random(w_var_true, drift_strength * 0.1)
            drift_flag = True
            drift_type = "train"
        else:
            drift_flag = False
            drift_type = "none"

        # Kantenexistenz
        A_t = (rng.random((n, n)) < p_true).astype(np.uint8)

        # Gewichte
        W_t = w_mean_true + rng.normal(0, np.sqrt(w_var_true), size=(n, n))
        W_t = np.clip(W_t, 0.0, 1.0).astype(np.float32)

        # Metadaten
        M_t = {
            "t": t,
            "phase": "train" if t < T_train else "predict",
            "drift": drift_flag,
            "drift_type": drift_type,
            "p_true": p_true,
            "F_true": (w_mean_true, w_var_true),
        }

        yield {"A": A_t, "W": W_t, "M": M_t}


# ================================================================
#  3. Generator: Drift während der Vorhersagephase
# ================================================================

def generate_stream_drift_prediction(
    n: int,
    T: int,
    T_train: int,
    drift_strength: float = 0.01
) -> Generator[Dict[str, object], None, None]:
    """
    Erzeugt einen Graphstream mit Drift während der Vorhersagephase.

    Parameters
    ----------
    n : int
        Anzahl der Knoten.

    T : int
        Anzahl der Zeitschritte.

    T_train : int
        Länge der Trainingsphase.

    drift_strength : float
        Stärke des Drifts.

    Yields
    ------
    dict
        Stream-Dictionary (A, W, M).
    """
    rng = np.random.default_rng()

    # Initiale Parameter
    p_true = rng.uniform(0.1, 0.9, size=(n, n)).astype(np.float32)
    w_mean_true = rng.uniform(0.2, 0.8, size=(n, n)).astype(np.float32)
    w_var_true = rng.uniform(0.01, 0.05, size=(n, n)).astype(np.float32)

    for t in range(T):

        # Drift nur während Vorhersage
        if t >= T_train:
            p_true = drift_random(p_true, drift_strength)
            w_mean_true = drift_random(w_mean_true, drift_strength)
            w_var_true = drift_random(w_var_true, drift_strength * 0.1)
            drift_flag = True
            drift_type = "predict"
        else:
            drift_flag = False
            drift_type = "none"

        # Kantenexistenz
        A_t = (rng.random((n, n)) < p_true).astype(np.uint8)

        # Gewichte
        W_t = w_mean_true + rng.normal(0, np.sqrt(w_var_true), size=(n, n))
        W_t = np.clip(W_t, 0.0, 1.0).astype(np.float32)

        # Metadaten
        M_t = {
            "t": t,
            "phase": "train" if t < T_train else "predict",
            "drift": drift_flag,
            "drift_type": drift_type,
            "p_true": p_true,
            "F_true": (w_mean_true, w_var_true),
        }

        yield {"A": A_t, "W": W_t, "M": M_t}
