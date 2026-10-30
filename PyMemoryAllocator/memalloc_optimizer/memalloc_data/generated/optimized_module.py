import numpy as np
import numba
import numpy as np
import time
from numba import njit
try:
    from qiskit_aer import Aer
except ImportError:
    from qiskit import Aer
from qiskit import QuantumCircuit
try:
    from qiskit import execute
except ImportError:

    @numba.njit
    def execute(circuit, backend):
        return backend.run(circuit)
from qiskit.circuit import Parameter


@njit
@numba.njit
def apply_pauli_ops(state, pauli):
    temp = state.copy()
    for i in range(len(pauli)):
        p = pauli[i]
        if p == 88:
            temp[i] = np.conj(temp[i])
        elif p == 89:
            temp[i] = -np.conj(temp[i])
    return temp


@njit
@numba.njit
def expectation_numba(state, coeffs, paulis):
    exp_val = 0.0
    for idx in range(len(coeffs)):
        coeff = coeffs[idx]
        pauli = paulis[idx]
        temp_state = apply_pauli_ops(state, pauli)
        exp_val += coeff * np.sum(np.abs(temp_state) ** 2)
    return exp_val


@numba.njit
def build_ansatz(num_qubits, depth):
    circuit = QuantumCircuit(num_qubits)
    theta = Parameter('θ')
    for layer in range(depth):
        for q in range(num_qubits):
            circuit.rx(theta, q)
        for q in range(num_qubits - 1):
            circuit.cx(q, q + 1)
    return circuit, theta


@numba.njit
def generate_hamiltonian(n_terms, n_qubits):
    coeffs = np.random.randn(n_terms)
    paulis = []
    for _ in range(n_terms):
        term = np.random.choice(['I', 'X', 'Y', 'Z'], size=n_qubits)
        paulis.append(np.array([ord(t) for t in term], dtype=np.int32))
    return coeffs, paulis


@numba.njit
def run_vqe():
    num_qubits = 6
    depth = 3
    n_terms = 50
    circuit, theta = build_ansatz(num_qubits, depth)
    coeffs, paulis = generate_hamiltonian(n_terms, num_qubits)
    backend = Aer.get_backend('statevector_simulator')
    theta_val = 0.1
    lr = 0.05
    start = time.perf_counter()
    for step in range(20):
        bound = circuit.assign_parameters({theta: theta_val})
        state = np.array(execute(bound, backend).result().get_statevector())
        exp_val = expectation_numba(state, coeffs, paulis)
        bound_grad = circuit.assign_parameters({theta: theta_val + 0.001})
        grad_state = np.array(execute(bound_grad, backend).result().
            get_statevector())
        grad = (expectation_numba(grad_state, coeffs, paulis) - exp_val
            ) / 0.001
        theta_val -= lr * grad
    end = time.perf_counter()
    print(f'Final theta: {theta_val:.4f}')
    print(f'Final energy: {exp_val:.6f}')
    print(f'Total runtime: {end - start:.3f} seconds')
