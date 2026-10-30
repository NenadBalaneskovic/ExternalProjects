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
# 1. Quantum feature map
# ============================================================

def feature_map(x):
    n = len(x)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(x[i], i)
    return qc


# ============================================================
# 2. Quantum neural network layer
# ============================================================

def qnn_layer(n_qubits, weights):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.rx(weights[i], i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


# ============================================================
# 3. Forward pass (Python-level hotspots)
# ============================================================

def forward_pass(x, weights):
    fm = feature_map(x)
    layer = qnn_layer(len(x), weights)

    qc = fm.compose(layer)

    backend = Aer.get_backend("statevector_simulator")
    state = execute(qc, backend).result().get_statevector()

    # Temporary array hotspot
    probs = np.abs(state)**2
    return np.sum(probs)


# ============================================================
# 4. Training loop
# ============================================================

def train_qnn():
    n_qubits = 5
    weights = np.random.randn(n_qubits)
    lr = 0.1

    start = time.perf_counter()

    for step in range(30):
        x = np.random.rand(n_qubits)

        out = forward_pass(x, weights)

        # Temporary array hotspot
        grad = np.random.randn(n_qubits)

        weights -= lr * grad

    end = time.perf_counter()

    print(f"Final weights: {weights}")
    print(f"Total runtime: {end - start:.3f} seconds")


if __name__ == "__main__":
    train_qnn()
