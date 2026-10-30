"""
qrng.py
-------

Quantum Random Number Generator (QRNG) based on fractional Schrödinger evolution.

This module depends on:
    - fractional_dynamics.py
    - qutip
    - numpy

Features:
    - Computational basis measurement
    - POVM measurement
    - Bitstring generation using fractional evolution
    - Basic randomness tests:
        * Frequency test
        * Autocorrelation
        * Shannon entropy
        * Min-entropy
        * Collision entropy
"""

import numpy as np
import qutip as qt

from fractional_dynamics import (
    generate_alpha_sequence,
    evolve_fractional
)


# ---------------------------------------------------------------------------
# Measurement Models
# ---------------------------------------------------------------------------

def measure_computational(psi: qt.Qobj) -> int:
    """
    Measure a qubit state in the computational basis.
    Returns 0 or 1.
    """
    probs = np.abs(psi.full())**2
    p0 = probs[0, 0]
    return 0 if np.random.rand() < p0 else 1


def measure_povm(psi: qt.Qobj) -> int:
    """
    POVM measurement using projectors |0><0| and |1><1|.
    Equivalent to computational basis measurement.
    """
    P0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
    p0 = (psi.dag() * P0 * psi).full()[0, 0].real
    return 0 if np.random.rand() < p0 else 1


# ---------------------------------------------------------------------------
# Bitstring Generation
# ---------------------------------------------------------------------------

def generate_bitstring(seed: int, N_steps: int, L: int) -> np.ndarray:
    """
    Generate a raw bitstring using fractional evolution and measurement.

    Steps:
        1. Generate αₖ sequence from seed
        2. Evolve initial state |0⟩ using fractional dynamics
        3. Measure final state
        4. Repeat L times
    """
    alpha_seq = generate_alpha_sequence(seed, N_steps)
    psi0 = qt.basis(2, 0)

    bits = []
    for _ in range(L):
        states = evolve_fractional(psi0, alpha_seq)
        psi_final = states[-1]
        bit = measure_computational(psi_final)
        bits.append(bit)

    return np.array(bits)


# ---------------------------------------------------------------------------
# Randomness Tests
# ---------------------------------------------------------------------------

def frequency_test(bits: np.ndarray):
    """
    Frequency test: returns (p0, p1).
    """
    p1 = bits.mean()
    p0 = 1 - p1
    return p0, p1


def autocorrelation(bits: np.ndarray, lag: int = 1):
    """
    Autocorrelation test for given lag.
    """
    if len(bits) <= lag:
        return 0.0
    return np.corrcoef(bits[:-lag], bits[lag:])[0, 1]


def shannon_entropy(bits: np.ndarray):
    """
    Shannon entropy H = -Σ p log2 p.
    """
    p1 = bits.mean()
    p0 = 1 - p1
    return -(p0*np.log2(p0 + 1e-12) + p1*np.log2(p1 + 1e-12))


def min_entropy(bits: np.ndarray):
    """
    Min-entropy H_min = -log2(max(p)).
    """
    p1 = bits.mean()
    p0 = 1 - p1
    return -np.log2(max(p0, p1))


def collision_entropy(bits: np.ndarray):
    """
    Collision entropy H2 = -log2(p0^2 + p1^2).
    """
    p1 = bits.mean()
    p0 = 1 - p1
    return -np.log2(p0**2 + p1**2)