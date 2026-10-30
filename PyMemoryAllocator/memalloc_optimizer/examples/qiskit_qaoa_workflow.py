import numpy as np
import time

# ------------------------------------------------------------
# Aer moved out of qiskit in v1.0 → try new location first
# ------------------------------------------------------------
try:
    from qiskit_aer import Aer
except ImportError:
    from qiskit import Aer

# ------------------------------------------------------------
# QuantumCircuit is still in qiskit
# ------------------------------------------------------------
from qiskit import QuantumCircuit

# ------------------------------------------------------------
# execute() was removed in Qiskit 1.x → provide fallback wrapper
# ------------------------------------------------------------
try:
    from qiskit import execute
except ImportError:
    def execute(circuit, backend):
        return backend.run(circuit)

# ------------------------------------------------------------
# Parameter is still in qiskit.circuit
# ------------------------------------------------------------
from qiskit.circuit import Parameter


# ============================================================
# 1. Build QAOA mixer + cost Hamiltonian
# ============================================================

def build_qaoa_layer(n_qubits, gamma, beta):
    qc = QuantumCircuit(n_qubits)

    # Cost Hamiltonian (ZZ interactions)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(2 * gamma, i + 1)
        qc.cx(i, i + 1)

    # Mixer Hamiltonian (RX rotations)
    for i in range(n_qubits):
        qc.rx(2 * beta, i)

    return qc


# ============================================================
# 2. Evaluate expectation value (Python-level hotspot)
# ============================================================

def evaluate_cost(state):
    # Temporary array hotspot
    probs = np.abs(state)**2
    return np.sum(probs)


# ============================================================
# 3. QAOA loop
# ============================================================

def run_qaoa():
    n_qubits = 6
    p = 3

    backend = Aer.get_backend("statevector_simulator")

    gamma = 0.1
    beta = 0.2

    start = time.perf_counter()

    for step in range(20):
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))

        for _ in range(p):
            qc.compose(build_qaoa_layer(n_qubits, gamma, beta), inplace=True)

        state = execute(qc, backend).result().get_statevector()

        cost = evaluate_cost(state)

        gamma -= 0.01 * cost
        beta -= 0.01 * cost

    end = time.perf_counter()

    print(f"Final gamma: {gamma:.4f}")
    print(f"Final beta: {beta:.4f}")
    print(f"Final cost: {cost:.6f}")
    print(f"Total runtime: {end - start:.3f} seconds")


if __name__ == "__main__":
    run_qaoa()
