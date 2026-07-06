# ldpc_codes.py
import numpy as np

def random_code(n, weight=3):
    """
    Lightweight LDPC code generator.
    Produces:
        H : parity-check matrix (n x n)
        G : generator matrix (n x n)
    """

    # Parity-check matrix H (sparse)
    H = np.zeros((n, n), dtype=int)
    rng = np.random.default_rng()

    for col in range(n):
        rows = rng.choice(n, size=weight, replace=False)
        H[rows, col] = 1

    # Ensure no all-zero rows
    for i in range(n):
        if np.sum(H[i]) == 0:
            j = rng.integers(0, n)
            H[i, j] = 1

    # Generator matrix G (identity for simplicity)
    G = np.eye(n, dtype=int)

    return H, G