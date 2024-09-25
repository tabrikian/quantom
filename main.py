from __future__ import annotations

import math

import numpy as np

NUMQUBITS = None


class Gate:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        # Matrix len must be power of 2
        self.size = round(np.log2(len(matrix)))

    @staticmethod
    def X() -> Gate:
        return Gate(np.array([[0, 1], [1, 0]], np.complex_))

    @staticmethod
    def R(k: int) -> Gate:
        return Gate(np.array([[1, 0], [0, np.exp(2j * np.pi / pow(2, k))]], np.complex_))

    @staticmethod
    def invR(k: int) -> Gate:  # for inverse QFT
        return Gate(np.array([[1, 0], [0, np.exp(-2j * np.pi / pow(2, k))]], np.complex_))

    @staticmethod
    def H() -> Gate:
        return Gate(np.array([[1, 1], [1, -1]], np.complex_) / np.sqrt(2))

    @staticmethod
    def Z() -> Gate:
        return Gate(np.array([[1, 0], [0, -1]], np.complex_))

    @staticmethod
    def S() -> Gate:
        return Gate(np.array([[1, 0], [0, 1j]], np.complex_))

    @staticmethod
    def T() -> Gate:
        return Gate(np.array([[1, 0], [0, np.exp(np.pi * 1j / 4)]], np.complex_))

    def exp_gates(U: Gate, n: int) -> list[Gate, ...]:
        powers_of_U = [U]
        for k in range(1, n):
            past_gate = powers_of_U[-1]
            new_gate = Gate(past_gate.matrix * past_gate.matrix)  # U**(2**k)
            powers_of_U.append(new_gate)
        return powers_of_U

    # @staticmethod
    def get_control_gate(U: Gate) -> Gate:
        n = pow(2, U.size)
        matrix = U.matrix
        matrix = np.concatenate((np.zeros((n, n)), matrix), axis=1)
        matrix = np.concatenate((np.concatenate((np.eye(n), np.zeros((n, n))), axis=1), matrix), axis=0)
        return Gate(matrix)

    def __repr__(self):
        return self.matrix.__repr__()


class QuantumComputer:
    @staticmethod
    def tensor_product_vector(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        result = np.array([], np.complex_)
        for i in v2:
            result = np.concatenate((result, i * v1))
        return result

    # Should be moved to new Class - Matrix class
    @staticmethod
    def tensor_product_matrix(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
        result = np.empty((len(mat1) * len(mat2), 0), np.complex_)
        for row in mat1:
            row_result = np.empty((0, len(mat2[0])), np.complex_)
            for val in row:
                row_result = np.concatenate((row_result, val * mat2), axis=0)
            result = np.concatenate((result, row_result), axis=1)
        return result

    # if init_state is None, set the circuit with 1 qbit and state |0>, else set the state to init_state
    # if init_state does not use all the qbits of the computer, set the first qbits as in init_state and set the other to |0>
    def __init__(self, num_of_qbits: int, init_state: np.ndarray = None):
        self.num_of_qbits = num_of_qbits
        if init_state is None:
            self.state = np.zeros(pow(2, num_of_qbits), dtype=np.complex_)
            self.state[0] = 1
        else:
            num_of_init_state_qbits = round(np.log2(len(init_state)))
            self.state = np.zeros(pow(2, num_of_qbits - num_of_init_state_qbits), dtype=np.complex_)
            self.state[0] = 1
            self.state = self.tensor_product_vector(init_state, self.state)

    def measure(self, qbit: int) -> int:
        # the qbit result is 0 in the state |i> if and only if i//qbit_option is even
        qbit_option = pow(2, self.num_of_qbits - qbit - 1)

        state_if_got0 = np.array([self.state[i] if (i // qbit_option) % 2 == 0 else 0 for i in range(len(self.state))],
                                 np.complex_)
        state_if_got1 = np.array([self.state[i] if (i // qbit_option) % 2 == 1 else 0 for i in range(len(self.state))],
                                 np.complex_)

        norm_state_if_got0 = np.linalg.norm(state_if_got0)
        norm_state_if_got1 = np.linalg.norm(state_if_got1)
        result = np.random.choice((0, 1), p=(norm_state_if_got0 ** 2, norm_state_if_got1 ** 2))

        self.state = state_if_got0 / norm_state_if_got0 if result == 0 else state_if_got1 / norm_state_if_got1
        return result

    def swap(self, i: int, j: int) -> None:
        if i == j:
            return

        qbit_value_i = pow(2, self.num_of_qbits - i - 1)
        qbit_value_j = pow(2, self.num_of_qbits - j - 1)
        new_state = np.zeros((len(self.state),), np.complex_)

        for n in range(len(self.state)):
            ith_bit = (n // qbit_value_i) % 2
            jth_bit = (n // qbit_value_j) % 2
            n_without_ij_bits = n - ith_bit * qbit_value_i - jth_bit * qbit_value_j
            n_swaped = n_without_ij_bits + ith_bit * qbit_value_j + jth_bit * qbit_value_i
            new_state[n_swaped] = self.state[n]

        self.state = new_state

    def apply_gate(self, gate: Gate, qbits: list[int, ...]) -> None:
        permutation = list(range(self.num_of_qbits))
        swap_queue = []
        # set the wanted qbits to the first qbits in the computer
        for i in range(len(qbits)):
            swap_queue.append((i, permutation[qbits[i]]))
            self.swap(i, permutation[qbits[i]])
            permutation[i], permutation[permutation[qbits[i]]] = permutation[permutation[qbits[i]]], permutation[i]

        # apply the gate on the first qbits
        extend_gate = self.tensor_product_matrix(gate.matrix, np.eye(pow(2, self.num_of_qbits - gate.size)))
        self.state = extend_gate.dot(self.state)

        # re-swap the qbits
        for i, j in reversed(swap_queue):
            self.swap(i, j)

    def apply_multiple_gates(self, gates: list[tuple[Gate, list[int, ...]], ...]) -> None:
        big_gate = np.eye(1)
        qbits = []
        for gate, gate_qbits in gates:
            qbits += list(gate_qbits)
            big_gate = self.tensor_product_matrix(big_gate, gate.matrix)
        self.apply_gate(Gate(big_gate), qbits)

    def __repr__(self):
        return "num of qbits: " + str(self.num_of_qbits) + ", state: " + str(self.state)


def Phase_Estimation(QC: QuantumComputer, U: Gate) -> list[int]:
    t = QC.num_of_qbits - U.size
    second_register = [t + i for i in range(QC.num_of_qbits - t)]

    # The first H gates can be done in parallel
    QC.apply_multiple_gates([(Gate.H(), [i]) for i in range(t)])

    # Apply Black Box
    for i, G in enumerate(U.exp_gates(t)):
        CG = G.get_control_gate()
        QC.apply_gate(CG, [t-1-i] + second_register)

    # Inverse QFT
    for i in range(t):
        for j in range(i, 0, -1):
            QC.apply_gate(Gate.invR(j + 1).get_control_gate(), [i - j, i])
        QC.apply_gate(Gate.H(), [i])

    # Measure everyone
    result = [QC.measure(i) for i in range(t)]
    result.reverse()
    return result


def main():
    U = None  # Input here the unitary matrix induced by gate C
    psi = None  # Input here the quantum state psi which is an eigenvector of U
    r = None  # Input here the accuracy parameter
    # t = math.ceil(math.log2(r))

    k = 5
    U = Gate(np.array([[1, 0], [0, np.exp(2j * np.pi / pow(2, k))]], np.complex_))
    psi = np.array([0, 1])
    r = 2 ** 4
    t = math.ceil(math.log2(r))

    global NUMQUBITS
    NUMQUBITS = t + round(math.log2(len(psi)))
    results = {}
    for _ in range(1000):
        QC = QuantumComputer(NUMQUBITS, psi)
        curr_result = tuple(Phase_Estimation(QC, U))
        if curr_result not in results.keys():
            results[curr_result] = 0
        results[curr_result] += 1
    print(results)
    return


if __name__ == "__main__":
    main()
    """t = 3
    for i in range(t):
        for j in range(i, 0, -1):
            print(i, j + 1, i - j)
            # QC.apply_gate(Gate.invR(j + 1).get_control_gate(), [i - j, i])
        print("H", i)
        # QC.apply_gate(Gate.H(), [i])"""
    """cmp = QuantumComputer(3)
    cmp.apply_multiple_gates(((Gate.Z(), (0,)), (Gate.H(), (1,))))
    print(cmp)

    cmp = QuantumComputer(3)
    cmp.apply_gate(Gate.Z(), (0,))
    print(cmp)
    cmp.apply_gate(Gate.H(), (1,))
    print(cmp)"""
