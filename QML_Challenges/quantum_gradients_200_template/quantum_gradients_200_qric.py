#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    nominal=circuit(weights)
    # QHACK #
    shift=np.pi/4
    erow1=np.zeros([5], dtype=np.float64)
    erow2=np.zeros([5], dtype=np.float64)

    # ehess=np.zeros([5, 5], dtype=np.float64)
    hess_norm=4*np.sin(shift)*np.sin(shift)
    grad_norm=2*np.sin(2*shift)
    for i in range(len(weights)): #row
        erow1[i]=1
        fwd_term=circuit(weights+2*shift*erow1)
        bkwd_term=circuit(weights-2*shift*erow1)
        gradient[i]=(fwd_term-bkwd_term)/grad_norm
        for j in range(i,len(weights)): #column
            erow2[j]=1
            if i==j:
                hessian[i,j]=(fwd_term+bkwd_term-2*nominal)/hess_norm
            else:
                first_term_1=circuit(weights+shift*erow1+shift*erow2)
                first_term_2=circuit(weights-shift*erow1+shift*erow2)
                sec_term_1=circuit(weights+shift*erow1-shift*erow2)
                sec_term_2=circuit(weights-shift*erow1-shift*erow2)
                hessian[i,j]=(first_term_1-first_term_2-sec_term_1+sec_term_2)/hess_norm
                hessian[j,i]=(first_term_1-first_term_2-sec_term_1+sec_term_2)/hess_norm
            erow2[j]=0
        erow1[i]=0
    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
