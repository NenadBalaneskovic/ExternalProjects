# ldpc_encoder.py
import numpy as np

def encode(G, bits):
    """
    Lightweight LDPC encoder.
    Encoding is simply:
        y = G @ bits  (mod 2)
    """
    bits = np.array(bits, dtype=int)
    y = (G @ bits) % 2
    return y