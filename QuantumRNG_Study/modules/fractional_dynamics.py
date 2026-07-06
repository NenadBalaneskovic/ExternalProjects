"""
fractional_dynamics.py
----------------------

Implements fractional Schrödinger evolution for a single qubit:

    - Fractional rotation angle θ(α)
    - Fractional unitary U(α)
    - Inverse unitary U⁻¹(α)
    - PRNG for fractional orders αₖ
    - Iterative evolution
    - Bloch vector extraction

Used by QRNG, QKD, fuzzy control, and QEC modules.
"""

import numpy as np
from scipy.special import gamma
import qutip as qt


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

hbar = 1.0
omega = 1.0
dt = 0.1

sx = qt.sigmax()
sz = qt.sigmaz()
id2 = qt.qeye(2)

H = (hbar * omega / 2.0) * sx


# ---------------------------------------------------------------------------
# Fractional rotation angle θ(α)
# ---------------------------------------------------------------------------

def theta(alpha: float, dt: float = dt, omega: float = omega) -> float:
    """
    Effective fractional rotation angle:
        θ(α) = (ω/2) * (dt**α) / Γ(α + 1)
    """
    return (omega / 2.0) * (dt ** alpha) / gamma(alpha + 1)


# ---------------------------------------------------------------------------
# Fractional unitary U(α)
# ---------------------------------------------------------------------------

def U_fractional(alpha: float) -> qt.Qobj:
    """
    Fractional evolution unitary:
        U(α) = exp(-i θ(α) σ_x)
    """
    th = theta(alpha)
    return (-1j * th * sx).expm()


# ---------------------------------------------------------------------------
# Inverse fractional unitary U⁻¹(α)
# ---------------------------------------------------------------------------

def U_inverse(alpha: float) -> qt.Qobj:
    """
    Inverse fractional unitary:
        U⁻¹(α) = exp(+i θ(α) σ_x)
    """
    th = theta(alpha)
    return (1j * th * sx).expm()


# ---------------------------------------------------------------------------
# PRNG for fractional orders αₖ
# ---------------------------------------------------------------------------

def generate_alpha_sequence(seed: int, N: int, low: float = 1.0, high: float = 2.0):
    """
    Generate fractional orders αₖ ∈ [low, high] using a reproducible PRNG.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=N)


# ---------------------------------------------------------------------------
# Iterative fractional evolution
# ---------------------------------------------------------------------------

def evolve_fractional(psi_init: qt.Qobj, alpha_seq: np.ndarray):
    """
    Apply U(αₖ) sequentially to initial state ψ₀.
    Returns list of states [ψ₀, ψ₁, ..., ψ_N].
    """
    states = [psi_init]
    psi = psi_init
    for a in alpha_seq:
        psi = U_fractional(a) * psi
        states.append(psi)
    return states


# ---------------------------------------------------------------------------
# Bloch vector extraction
# ---------------------------------------------------------------------------

def bloch_components(state: qt.Qobj):
    """
    Return (x, y, z) Bloch components of a qubit state.
    """
    vec = qt.bloch_vector(state)
    return vec[0], vec[1], vec[2]