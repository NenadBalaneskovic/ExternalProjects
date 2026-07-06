"""
qkd_fractional.py
-----------------

Fractional-controlled Quantum Key Distribution (QKD) protocol.

Depends on:
    - fractional_dynamics.py
    - qrng.py
    - qutip
    - numpy

Features:
    - Shared αₖ generation from seed
    - Alice encoding using fractional evolution
    - Bob decoding using inverse fractional evolution
    - Depolarizing noise model
    - Intercept-resend attack model
    - Partial-knowledge attack model
    - QKD session runner
    - QBER computation
"""

import numpy as np
import qutip as qt

from fractional_dynamics import (
    generate_alpha_sequence,
    evolve_fractional,
    U_inverse,
    U_fractional
)

from qrng import measure_computational


# ---------------------------------------------------------------------------
# Alice Encoding
# ---------------------------------------------------------------------------

def alice_encode_bit(m: int, alpha_seq: np.ndarray) -> qt.Qobj:
    """
    Alice encodes classical bit m ∈ {0,1} using fractional evolution.
    """
    psi_init = qt.basis(2, m)
    states = evolve_fractional(psi_init, alpha_seq)
    return states[-1]


# ---------------------------------------------------------------------------
# Bob Decoding
# ---------------------------------------------------------------------------

def bob_decode(psi_received: qt.Qobj, alpha_seq: np.ndarray) -> int:
    """
    Bob applies inverse fractional evolution and measures.
    """
    psi = psi_received
    for a in reversed(alpha_seq):
        psi = U_inverse(a) * psi
    return measure_computational(psi)


# ---------------------------------------------------------------------------
# Noise Models
# ---------------------------------------------------------------------------

def depolarize(psi: qt.Qobj, p: float) -> qt.Qobj:
    """
    Depolarizing channel:
        ρ → (1-p)ρ + p I/2
    """
    rho = psi * psi.dag()
    return (1 - p) * rho + p * (qt.qeye(2) / 2)


def intercept_resend(psi: qt.Qobj) -> qt.Qobj:
    """
    Eve measures and resends her guess.
    """
    bit = measure_computational(psi)
    return qt.basis(2, bit)


def eve_partial_inverse(psi: qt.Qobj, alpha_seq: np.ndarray, knowledge_prob=0.3):
    """
    Eve knows αₖ with probability knowledge_prob.
    She attempts partial inverse evolution.
    """
    psi_eve = psi
    for a in reversed(alpha_seq):
        if np.random.rand() < knowledge_prob:
            psi_eve = U_inverse(a) * psi_eve
        else:
            wrong_a = a + np.random.normal(0, 0.2)
            wrong_a = np.clip(wrong_a, 1.0, 2.0)
            psi_eve = U_inverse(wrong_a) * psi_eve
    return psi_eve


# ---------------------------------------------------------------------------
# QKD Session Runner
# ---------------------------------------------------------------------------

def run_qkd_session(seed: int, N_steps: int, L: int,
                    noise_p=0.0, intercept=False, partial=False, knowledge_prob=0.3):
    """
    Run a full fractional-controlled QKD session.

    Parameters:
        seed          → shared seed for αₖ
        N_steps       → number of fractional evolution steps
        L             → number of bits to generate
        noise_p       → depolarizing noise probability
        intercept     → Eve performs intercept-resend
        partial       → Eve performs partial inverse attack
        knowledge_prob→ probability Eve knows αₖ

    Returns:
        (K_A, K_B) raw keys
    """
    alpha_seq = generate_alpha_sequence(seed, N_steps)
    K_A = []
    K_B = []

    for _ in range(L):
        m = np.random.randint(0, 2)
        psi_A = alice_encode_bit(m, alpha_seq)

        # Channel
        if intercept:
            psi_channel = intercept_resend(psi_A)
        elif partial:
            psi_channel = eve_partial_inverse(psi_A, alpha_seq, knowledge_prob)
        else:
            psi_channel = psi_A

        if noise_p > 0:
            psi_channel = depolarize(psi_channel, noise_p)

        b = bob_decode(psi_channel, alpha_seq)

        K_A.append(m)
        K_B.append(b)

    return np.array(K_A), np.array(K_B)


# ---------------------------------------------------------------------------
# QBER
# ---------------------------------------------------------------------------

def qber(KA: np.ndarray, KB: np.ndarray) -> float:
    """
    Quantum Bit Error Rate:
        QBER = (# mismatches) / len(K)
    """
    return np.mean(KA != KB)