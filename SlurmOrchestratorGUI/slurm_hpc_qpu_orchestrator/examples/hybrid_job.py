"""
Hybrid HPC–QPU workload.
Used to test Slurm Orchestrator classification.
"""

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator
from qiskit import QuantumCircuit

def run():
    # Classical initialization
    params = np.random.randn(4)

    # Quantum backend
    service = QiskitRuntimeService()
    estimator = Estimator()

    # Simple circuit
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)

    # Hybrid optimization loop
    for _ in range(50):
        value = estimator.run(qc, params).result().values[0]
        params = params - 0.1 * value  # classical update

    return params

if __name__ == "__main__":
    print(run())