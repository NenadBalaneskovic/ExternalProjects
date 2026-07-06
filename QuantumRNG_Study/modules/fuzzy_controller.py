# modules/fuzzy_controller.py

import numpy as np

# ---------------------------------------------------------
# Triangular membership function
# ---------------------------------------------------------

def tri(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    else:
        return 1.0 if x == b else 0.0


# ---------------------------------------------------------
# Fuzzy sets (triangular)
# ---------------------------------------------------------

noise_sets = {
    "low":     (0.0, 0.0, 0.3),
    "medium":  (0.2, 0.5, 0.8),
    "high":    (0.6, 1.0, 1.0)
}

qber_sets = {
    "low":     (0.0, 0.0, 0.1),
    "medium":  (0.05, 0.2, 0.35),
    "high":    (0.3, 0.5, 0.5)
}

entropy_sets = {
    "low":     (0.0, 0.0, 0.3),
    "medium":  (0.2, 0.5, 0.8),
    "high":    (0.7, 1.0, 1.0)
}

alpha_sets = {
    "weak":     (0.0, 0.0, 0.3),
    "moderate": (0.2, 0.5, 0.8),
    "strong":   (0.7, 1.0, 1.0)
}

basis_sets = {
    "none":     (0.0, 0.0, 0.3),
    "slight":   (0.2, 0.5, 0.8),
    "strong":   (0.7, 1.0, 1.0)
}

qec_sets = {
    "weak":     (0.0, 0.0, 0.3),
    "medium":   (0.2, 0.5, 0.8),
    "strong":   (0.7, 1.0, 1.0)
}


# ---------------------------------------------------------
# Pure-Python fuzzy inference engine (Mamdani-style)
# ---------------------------------------------------------

def run_fuzzy_controller(noise_val, qber_val, entropy_val):

    # Membership values
    noise_m = {k: tri(noise_val, *v) for k, v in noise_sets.items()}
    qber_m  = {k: tri(qber_val,  *v) for k, v in qber_sets.items()}
    entr_m  = {k: tri(entropy_val, *v) for k, v in entropy_sets.items()}

    # Rule outputs (Mamdani min)
    alpha_out = []
    basis_out = []
    qec_out   = []

    # Rules
    alpha_out.append(("strong",  min(noise_m["high"], entr_m["low"])))
    qec_out.append(("medium",    qber_m["medium"]))
    qec_out.append(("strong",    qber_m["high"]))
    basis_out.append(("strong",  qber_m["high"]))
    alpha_out.append(("weak",    min(noise_m["low"], entr_m["high"])))
    alpha_out.append(("moderate", noise_m["medium"]))
    basis_out.append(("slight",  entr_m["medium"]))

    # Defuzzification (centroid)
    def defuzz(sets, outputs):
        xs = np.linspace(0, 1, 400)
        ys = np.zeros_like(xs)
        for term, strength in outputs:
            a, b, c = sets[term]
            ys = np.maximum(ys, strength * np.array([tri(x, a, b, c) for x in xs]))
        return float(np.sum(xs * ys) / np.sum(ys)) if np.sum(ys) > 0 else 0.0

    return {
        "alpha_var":    defuzz(alpha_sets, alpha_out),
        "basis_shift":  defuzz(basis_sets, basis_out),
        "qec_strength": defuzz(qec_sets, qec_out)
    }