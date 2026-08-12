"""
Pure classical HPC workload.
Used to test Slurm Orchestrator classification.
"""

import numpy as np
import scipy.linalg as la

def run():
    # Heavy classical linear algebra
    A = np.random.randn(2000, 2000)
    B = la.inv(A)
    return np.sum(B)

if __name__ == "__main__":
    print(run())