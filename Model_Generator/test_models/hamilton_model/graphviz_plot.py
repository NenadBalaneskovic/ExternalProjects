"""
graphviz_plot.py
====================

Graphviz-Visualisierung des Hamilton-Zyklus.
Modular, ohne globale Variablen, kompatibel mit PosteriorState
und der gesamten Pipeline.
"""

from graphviz import Digraph
import numpy as np
from typing import List


def plot_hamilton_graphviz(
    cycle: np.ndarray,
    p_post: np.ndarray,
    w_mean_post: np.ndarray,
    w_var_post: np.ndarray,
    title: str = "Hamilton-Zyklus (Graphviz)",
    highlight_stable: bool = True,
    λ1: float = 2.0,
    λ2: float = 1.0
) -> Digraph:
    """
    Erzeugt einen Graphviz-Plot des Hamilton-Zyklus.

    Parameters
    ----------
    cycle : np.ndarray
        Hamilton-Zyklus als Permutation der Knoten.

    p_post : np.ndarray
        Posterior-Existenzwahrscheinlichkeiten.

    w_mean_post : np.ndarray
        Posterior-Gewichtserwartungen.

    w_var_post : np.ndarray
        Posterior-Gewichtsvarianzen.

    title : str
        Titel des Graphviz-Plots.

    highlight_stable : bool
        Falls True, werden die stabilsten Kanten farblich hervorgehoben.

    λ1, λ2 : float
        Gewichtungsparameter für Score-Berechnung.

    Returns
    -------
    Digraph
        Graphviz-Diagramm des Hamilton-Zyklus.
    """

    dot = Digraph(comment=title)
    dot.attr(rankdir="LR")  # horizontaler Plot

    n = len(cycle)

    # Knoten hinzufügen
    for node in cycle:
        dot.node(str(node), str(node))

    # Kanten hinzufügen
    for i in range(n - 1):
        u = cycle[i]
        v = cycle[i + 1]

        # Posterior-Score der Kante
        score = λ1 * p_post[u, v] - λ2 * w_var_post[u, v]

        # Farbe bestimmen
        if highlight_stable:
            if score > 1.5:
                color = "green"
                penwidth = "3"
            elif score > 1.0:
                color = "blue"
                penwidth = "2"
            else:
                color = "gray"
                penwidth = "1"
        else:
            color = "black"
            penwidth = "1"

        # Kantenlabel: Gewicht
        label = f"{w_mean_post[u, v]:.2f}"

        dot.edge(str(u), str(v), label=label, color=color, penwidth=penwidth)

    return dot
