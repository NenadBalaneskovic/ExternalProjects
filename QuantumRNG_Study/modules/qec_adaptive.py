"""
qec_adaptive.py
----------------

Adaptive Quantum Error Correction (QEC) for fractional–fuzzy QKD.
Python 3.12 compatible version using the modern `ldpc` library.

Features:
    - Hamming(7,4) encoding/decoding
    - LDPC encoding/decoding (belief propagation)
    - Adaptive QEC wrapper selecting Hamming or LDPC
    - Privacy amplification using SHA3-256
"""

import numpy as np
from Crypto.Hash import SHA3_256

# Modern LDPC library (Python 3.12 compatible)
from ldpc.codes import random_code
from ldpc.encoder import encode
from ldpc.decoder import decode_bp

from fuzzy_controller import run_fuzzy_controller
from qkd_fractional import qber


# ---------------------------------------------------------------------------
# Hamming(7,4) Code
# ---------------------------------------------------------------------------

H_hamming = np.array([
    [1,0,1,0,1,0,1],
    [0,1,1,0,0,1,1],
    [0,0,0,1,1,1,1]
])

G_hamming = np.array([
    [1,0,0,0,1,1,0],
    [0,1,0,0,1,0,1],
    [0,0,1,0,0,1,1],
    [0,0,0,1,1,0,1]
])


def hamming_encode(block4):
    return (block4 @ G_hamming) % 2


def hamming_syndrome(code7):
    return (H_hamming @ code7) % 2


def hamming_correct(code7):
    s = hamming_syndrome(code7)
    if np.all(s == 0):
        return code7
    for i in range(7):
        if np.all(H_hamming[:, i] == s):
            code7[i] ^= 1
            break
    return code7


def reconcile_hamming(KA, KB):
    """
    Reconcile Bob's key KB to match Alice's KA using Hamming(7,4).
    KA and KB must be multiples of 4 bits.
    """
    KA_blocks = KA.reshape(-1, 4)
    KB_blocks = KB.reshape(-1, 4)

    corrected_bits = []

    for a_block, b_block in zip(KA_blocks, KB_blocks):
        codeA = hamming_encode(a_block)
        codeB = hamming_encode(b_block)

        corrected_codeB = hamming_correct(codeB)
        corrected_bits.extend(corrected_codeB[:4])

    return np.array(corrected_bits)


# ---------------------------------------------------------------------------
# LDPC QEC (Python 3.12 compatible)
# ---------------------------------------------------------------------------

def ldpc_reconcile(KA, KB, strength):
    """
    LDPC reconciliation using the modern `ldpc` library.
    """

    # Choose LDPC size based on fuzzy strength
    if strength < 0.33:
        n = 64
    elif strength < 0.66:
        n = 128
    else:
        n = 256

    # Generate random LDPC code
    H, G = random_code(n, weight=3)

    # Pad keys
    pad_len = n - len(KA)
    KA_pad = np.concatenate([KA, np.zeros(pad_len, dtype=int)])
    KB_pad = np.concatenate([KB, np.zeros(pad_len, dtype=int)])

    # Encode
    yA = encode(G, KA_pad)
    yB = encode(G, KB_pad)

    # Decode using belief propagation
    yB_corr = decode_bp(H, yB, max_iter=50)

    # Extract corrected bits
    return yB_corr[:len(KA)]


# ---------------------------------------------------------------------------
# Adaptive QEC Wrapper
# ---------------------------------------------------------------------------

def adaptive_qec(KA, KB, qec_strength):
    """
    Use fuzzy output to choose QEC method.
    """
    if qec_strength < 0.33:
        print("Using weak QEC: Hamming(7,4)")
        L = len(KA) - (len(KA) % 4)
        return reconcile_hamming(KA[:L], KB[:L])
    else:
        print("Using LDPC QEC (Python 3.12 compatible)")
        return ldpc_reconcile(KA, KB, qec_strength)


# ---------------------------------------------------------------------------
# Privacy Amplification
# ---------------------------------------------------------------------------

def privacy_amplification(K_bits):
    """
    Apply SHA3-256 to produce final shared key K.
    """
    bitstring = "".join(str(b) for b in K_bits)
    h = SHA3_256.new()
    h.update(bitstring.encode())
    return h.hexdigest()