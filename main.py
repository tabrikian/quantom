from __future__ import annotations
import numpy as np


class Gate:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
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
        return Gate(np.array([[1, 0], [0, np.exp(np.pi*1j/4)]], np.complex_))

    @staticmethod
    # for now using iterated squaring, need to verify with Yotam
    def power_gates(U: Gate, n: int) -> tuple[Gate, ...]:
        powers_of_U = [U]
        for k in range(1, n):
            past_gate = powers_of_U[-1]
            new_gate = Gate(past_gate.matrix * past_gate.matrix)  # U**(2**k)
            powers_of_U.append(new_gate)
        return tuple(powers_of_U)

    @staticmethod
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
            self.state = self.tensor_product_vector(self.state, init_state)

    def measure(self, qbit: int) -> int:
        # the qbit result is 0 in the state |i> if and only if i//qbit_option is even
        qbit_option = pow(2, self.num_of_qbits - qbit - 1)

        state_if_got0 = np.array([self.state[i] if (i // qbit_option) % 2 == 0 else 0 for i in range(len(self.state))], np.complex_)
        state_if_got1 = np.array([self.state[i] if (i // qbit_option) % 2 == 1 else 0 for i in range(len(self.state))], np.complex_)

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

    def apply_gate(self, gate: Gate, qbits: tuple[int, ...]) -> None:
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

    def apply_multiple_gates(self, gates: tuple[tuple[Gate, tuple[int, ...]], ...]) -> None:
        big_gate = np.eye(1)
        qbits = []
        for gate, gate_qbits in gates:
            qbits += list(gate_qbits)
            big_gate = self.tensor_product_matrix(big_gate, gate.matrix)
        self.apply_gate(Gate(big_gate), tuple(qbits))

    def __repr__(self):
        return "num of qbits: " + str(self.num_of_qbits) + ", state: " + str(self.state)


if __name__ == "__main__":
    cmp = QuantumComputer(3)
    cmp.apply_multiple_gates(((Gate.Z(), (0,)), (Gate.H(), (1,))))
    print(cmp)

    cmp = QuantumComputer(3)
    cmp.apply_gate(Gate.Z(), (0,))
    print(cmp)
    cmp.apply_gate(Gate.H(), (1,))
    print(cmp)
