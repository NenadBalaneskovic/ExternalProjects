"""
Pure quantum workload.
Used to test Slurm Orchestrator classification.
"""

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit

def run():
    service = QiskitRuntimeService()  # QPU session
    sampler = Sampler()

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    result = sampler.run(qc).result()
    return result

if __name__ == "__main__":
    print(run())