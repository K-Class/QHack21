#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #
    num_qubits = len(H.wires)
    num_param_sets = (2 ** num_qubits) - 1
    params = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(num_param_sets, 3))
    saved_params = []

    dev = qml.device("default.qubit", wires=num_qubits)

    # circuit from vqe-100
    def variational_ansatz(params, wires):
        n_qubits = len(wires)
        n_rotations = len(params)

        if n_rotations > 1:
            n_layers = n_rotations // n_qubits
            n_extra_rots = n_rotations - n_layers * n_qubits

            # Alternating layers of unitary rotations on every qubit followed by a
            # ring cascade of CNOTs.
            for layer_idx in range(n_layers):
                layer_params = params[layer_idx * n_qubits : layer_idx * n_qubits + n_qubits, :]
                qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
                qml.broadcast(qml.CNOT, wires, pattern="ring")

            # There may be "extra" parameter sets required for which it's not necessarily
            # to perform another full alternating cycle. Apply these to the qubits as needed.
            extra_params = params[-n_extra_rots:, :]
            extra_wires = wires[: n_qubits - 1 - n_extra_rots : -1]
            qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
        else:
            # For 1-qubit case, just a single rotation to the qubit
            qml.Rot(*params[0], wires=wires[0])

    #print("H",H)
    # find ground state
    cost0 = qml.ExpvalCost(variational_ansatz, H, dev)

    #opt = qml.GradientDescentOptimizer(0.1)
    opt = qml.AdamOptimizer(0.1)

    #print(H.wires)

    for i in range(400):
        #if i % 10: print(f"step {i}, E_0 {cost0(params)}")
        params = opt.step(cost0, params)  

    energies[0] = cost0(params)
    saved_params.append(params)
    #print(energies[0],cost0(params))

    # function for overlaps
    qml.enable_tape()

    @qml.qnode(dev)
    def get_state(params):
        variational_ansatz(params, dev.wires)
        return qml.state()
     
    overlap_state1 = get_state(params)
    overlap_herm1 = np.outer(overlap_state1.conj(), overlap_state1)
    #print("psi_0",overlap_state1)
    #print(overlap_herm1)

    # find the first excited
    params = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(num_param_sets, 3))
    a = 2 # big number to enforce orthogonality
    overlap_Ham = qml.Hamiltonian(coeffs=[a,], observables=[qml.Hermitian(overlap_herm1,dev.wires),])
    #print("a|psi_0><psi_0",overlap_Ham,overlap_Ham.ops)
    H1 = H + overlap_Ham
    #print("H1",H1)
    cost = qml.ExpvalCost(variational_ansatz, H1, dev)# + qml.ExpvalCost(variational_ansatz, overlap_Ham, dev)
    #print(cost(saved_params[0]),a+energies[0],a+cost0(saved_params[0]))

    for i in range(400):
        #if i % 10: print(f"step {i}, E_1 {cost0(params)}, cost {cost(params)}")
        params = opt.step(cost, params)  

    energies[1] = cost0(params)
    saved_params.append(params)
    #print(energies[1],cost(params))

    overlap_state2 = get_state(params)
    overlap_herm2 = np.outer(overlap_state2.conj(), overlap_state2)
    #print("|psi_1>",overlap_state2)
    #print(overlap_herm2)

    # find the second excited    
    params = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(num_param_sets, 3))
    b = 2
    overlap_Ham = qml.Hamiltonian(coeffs=[a,b], observables=[qml.Hermitian(overlap_herm1,dev.wires),qml.Hermitian(overlap_herm2,dev.wires)])
    #print("a|psi_0><psi_0|+b|psi_1><psi_1",overlap_Ham,overlap_Ham.ops)
    H2 = H + overlap_Ham
    #print("H2",H2)
    cost = qml.ExpvalCost(variational_ansatz, H2, dev)# + qml.ExpvalCost(variational_ansatz, overlap_Ham, dev)

    for i in range(400):
        #if i % 10: print(f"step {i}, E_2 {cost0(params)}, cost {cost(params)}")
        params = opt.step(cost, params)  

    energies[2] = cost0(params)
    saved_params.append(params)

    # QHACK #

    return ",".join([str(E) for E in energies])


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
