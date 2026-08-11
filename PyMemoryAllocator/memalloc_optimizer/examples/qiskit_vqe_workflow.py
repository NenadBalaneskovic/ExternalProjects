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
# 1. Build a parameterized ansatz
# ============================================================

def build_ansatz(num_qubits, depth):
    circuit = QuantumCircuit(num_qubits)
    theta = Parameter("θ")

    for layer in range(depth):
        for q in range(num_qubits):
            circuit.rx(theta, q)
        for q in range(num_qubits - 1):
            circuit.cx(q, q + 1)

    return circuit, theta


# ============================================================
# 2. Generate synthetic Hamiltonian
# ============================================================

def generate_hamiltonian(n_terms, n_qubits):
    coeffs = np.random.randn(n_terms)

    paulis = []
    for _ in range(n_terms):
        term = np.random.choice(["I", "X", "Y", "Z"], size=n_qubits)
        paulis.append("".join(term))

    return coeffs, paulis


# ============================================================
# 3. Evaluate expectation value
# ============================================================

def expectation_value(circuit, theta_value, coeffs, paulis):
    backend = Aer.get_backend("statevector_simulator")

    param = next(iter(circuit.parameters))
    bound_circuit = circuit.assign_parameters({param: theta_value})

    result = execute(bound_circuit, backend).result()

    # Convert Statevector → NumPy array (critical fix)
    state = np.array(result.get_statevector())

    exp_val = 0.0

    for coeff, pauli in zip(coeffs, paulis):
        temp_state = state.copy()
        for i, p in enumerate(pauli):
            if p == "X":
                temp_state[i] = np.conj(temp_state[i])
            elif p == "Y":
                temp_state[i] = -np.conj(temp_state[i])
        exp_val += coeff * np.sum(np.abs(temp_state) ** 2)

    return exp_val


# ============================================================
# 4. Simple VQE loop
# ============================================================

def run_vqe():
    num_qubits = 6
    depth = 3
    n_terms = 50

    circuit, theta = build_ansatz(num_qubits, depth)
    coeffs, paulis = generate_hamiltonian(n_terms, num_qubits)

    theta_val = 0.1
    lr = 0.05

    start = time.perf_counter()

    for step in range(20):
        exp_val = expectation_value(circuit, theta_val, coeffs, paulis)
        grad = (expectation_value(circuit, theta_val + 1e-3, coeffs, paulis) - exp_val) / 1e-3
        theta_val -= lr * grad

    end = time.perf_counter()

    print(f"Final theta: {theta_val:.4f}")
    print(f"Final energy: {exp_val:.6f}")
    print(f"Total runtime: {end - start:.3f} seconds")

if __name__ == "__main__":
    run_vqe()
