# ldpc_decoder.py
import numpy as np

def decode_bp(H, y, max_iter=50):
    """
    Lightweight belief propagation LDPC decoder.
    Hard-decision BP:
        - variable nodes send bits
        - check nodes enforce parity
        - iterate until convergence
    """

    y = np.array(y, dtype=int)
    n = len(y)
    x = y.copy()

    # Precompute check node connections
    checks = [np.where(H[i] == 1)[0] for i in range(H.shape[0])]
    vars_ = [np.where(H[:, j] == 1)[0] for j in range(H.shape[1])]

    for _ in range(max_iter):

        # Check node update
        for i, cn in enumerate(checks):
            if len(cn) == 0:
                continue
            parity = np.sum(x[cn]) % 2
            # If parity is wrong, flip one variable
            if parity != 0:
                j = cn[0]
                x[j] ^= 1

        # Variable node update
        for j, vn in enumerate(vars_):
            if len(vn) == 0:
                continue
            # Majority vote from connected checks
            votes = []
            for i in vn:
                cn = checks[i]
                votes.append(np.sum(x[cn]) % 2)
            if len(votes) > 0:
                x[j] = 1 if np.sum(votes) > len(votes)/2 else 0

        # Check if syndrome is zero
        syndrome = (H @ x) % 2
        if np.sum(syndrome) == 0:
            break

    return x